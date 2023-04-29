#!/usr/bin/env python3
# 自动起飞: https://docs.px4.io/main/en/ros/mavros_offboard_python.html
from offboard import MavrosOffboardPosctl
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from mavros_msgs.msg import ExtendedState, State
from mavros_msgs.srv import (
    ParamGet,
    ParamSet,
    CommandBool,
    CommandBoolRequest,
    SetMode,
    SetModeRequest,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from pymavlink import mavutil
from std_msgs.msg import Header, Bool, Float32
from threading import Thread
import numpy as np
import torch
from A2C import A2C
import matplotlib.pyplot as plt
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Quaternion, PointStamped, Point
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import time


class TrainSession(MavrosOffboardPosctl):
    def __init__(
        self,
        n_updates=1000,
        n_steps_per_update=128,
        gamma=0.999,
        lam=0.95,  # hyperparameter for GAE
        ent_coef=0.01,  # coefficient for the entropy bonus (to encourage exploration)
        actor_lr=0.001,
        critic_lr=0.005,
        use_cuda=False,
    ):
        rospy.init_node("offboard_node")

        # environment hyperparams
        self.n_updates = n_updates
        self.n_steps_per_update = n_steps_per_update
        self.local_position = PoseStamped()

        # agent hyperparams
        self.gamma = gamma
        self.lam = lam  # hyperparameter for GAE
        self.ent_coef = (
            ent_coef  # coefficient for the entropy bonus (to encourage exploration)
        )

        # Note: the actor has a slower learning rate so that the value targets become
        # more stationary and are theirfore easier to estimate for the critic

        # set the device
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.obs = np.zeros((3, 480, 640))
        self.rewards = 0
        self.terminated = False
        self.score = 0
        self.ready = False
        self.action = 0

        self.actions = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, -1],
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [-1, 0, 0],
            ],
            dtype=np.float64,
        )
        # init the agent
        obs_shape = self.obs.shape
        action_shape = len(self.actions)
        self.agent = A2C(obs_shape, action_shape, self.device, critic_lr, actor_lr, 0.3)

        self.critic_losses = []
        self.actor_losses = []
        self.entropies = []
        self.return_queue = []

        self.bridge = CvBridge()

        self.extended_state = ExtendedState()
        self.imu_data = Imu()
        self.local_position = PoseStamped()
        self.state = State()
        self.rover_pos = PointStamped()  # not used
        self.pos = PoseStamped()
        self.radius = 0.1
        
        self.sub_topics_ready = {
            key: False
            for key in [
                "ext_state",
                "local_pos",
                "state",
                "imu",
            ]
        }

        # ROS services
        service_timeout = 60
        rospy.loginfo("waiting for ROS services")
        try:
            rospy.loginfo("waiting for param get")
            rospy.wait_for_service("mavros/param/get", service_timeout)
            rospy.loginfo("waiting for cmd arming")
            rospy.wait_for_service("mavros/cmd/arming", service_timeout)
            rospy.loginfo("waiting for set mode")
            rospy.wait_for_service("mavros/set_mode", service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException as e:
            rospy.logerr("failed to connect to services")

        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.get_param_srv = rospy.ServiceProxy("mavros/param/get", ParamGet)
        self.set_param_srv = rospy.ServiceProxy("mavros/param/set", ParamSet)

        # ROS subscribers
        # self.rov_pos_sub = rospy.Subscriber(
        #     "rover/point", PointStamped, self.rover_pos_callback
        # )
        self.rov_pos_sub = rospy.Subscriber(
            "/qualisys/rover/odom", Odometry, self.rover_pos_callback
        )
        self.ext_state_sub = rospy.Subscriber(
            "mavros/extended_state", ExtendedState, self.extended_state_callback
        )
        self.imu_data_sub = rospy.Subscriber(
            "mavros/imu/data", Imu, self.imu_data_callback
        )
        self.local_pos_sub = rospy.Subscriber(
            "mavros/local_position/pose", PoseStamped, self.local_position_callback
        )
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_callback)
        self.camera_info_sub = rospy.Subscriber(
            "/drone/camera/color/camera_info", CameraInfo, self.camera_info_callback
        )
        self.camera_sub = rospy.Subscriber(
            "/drone/camera/color/image_raw", Image, self.image_callback, queue_size=1
        )
        self.reward_sub = rospy.Subscriber(
            "/inst_score", Float32, self.reward_callback, queue_size=1
        )
        self.return_sub = rospy.Subscriber(
            "/score", Float32, self.return_callback, queue_size=1
        )
        self.finished_sub = rospy.Subscriber(
            "/rover/finished", Bool, self.finished_callback
        )
        self.ready_pub = rospy.Publisher("ready", Bool, queue_size=10)
        self.pos_setpoint_pub = rospy.Publisher(
            "mavros/setpoint_position/local", PoseStamped, queue_size=1
        )

        # send setpoints in seperate thread to better prevent failsafe
        self.pos_thread = Thread(target=self.send_pos, args=())

    def camera_info_callback(self, msg: CameraInfo):
        """Callback from camera projetion"""
        self.camera_info = msg

    def image_callback(self, msg):
        if self.camera_info is None:
            rospy.logerr("no camera info")
            return

        obs = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.obs = np.array(obs).transpose((2, 0, 1))  # 3*480*640

    def reward_callback(self, msg):
        self.rewards = msg.data

    def return_callback(self, msg):
        self.score = msg.data

    def finished_callback(self, msg):
        self.terminated = msg.data
        if self.terminated:
            self.return_queue.append(self.score)
            self.reset()  # auto reset

    def wait_for_offborad(self):
        last_req = rospy.Time.now()
        rate = rospy.Rate(10)  # 10 Hz
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = "OFFBOARD"
        while not rospy.is_shutdown():
            if self.state.mode != "OFFBOARD":
                if (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    rospy.loginfo_throttle(10, "waiting for offborad")
                    continue
                if self.set_mode_client.call(offb_set_mode).mode_sent == True:
                    rospy.loginfo("offboard mode confirmed")
                    break
                else:
                    rospy.loginfo("set mode failed, retry")

            rate.sleep()
            last_req = rospy.Time.now()

    def wait_for_takeoff(self):
        last_req = rospy.Time.now()
        rate = rospy.Rate(10)  # 10 Hz
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        while not rospy.is_shutdown():
            if not self.state.armed:
                if (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    rospy.loginfo_throttle(10, "waiting for takeoff")
                    continue
                if self.arming_client.call(arm_cmd).success == True:
                    rospy.loginfo("Vehicle armed")
                    break
                else:
                    rospy.loginfo("arm call failed, retry")
            else:
                rospy.loginfo("Vehicle armed")

            last_req = rospy.Time.now()
            rate.sleep()

    def setup_env(self):
        # make sure the simulation is ready to start the mission
        self.ready_pub.publish(False)

        rospy.logwarn("waiting for landed state")
        self.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND)

        rospy.logwarn("setting parameters")
        self.set_param("EKF2_AID_MASK", 24, timeout=30, is_integer=True)
        self.set_param("EKF2_HGT_MODE", 3, timeout=5, is_integer=True)
        self.set_param("EKF2_EV_DELAY", 0.0, timeout=5, is_integer=False)
        self.set_param("MPC_XY_VEL_MAX", 1.0, timeout=5, is_integer=False)
        self.set_param("MC_YAWRATE_MAX", 60.0, timeout=5, is_integer=False)
        self.set_param("MIS_TAKEOFF_ALT", 1.0, timeout=5, is_integer=False)
        self.set_param("NAV_MC_ALT_RAD", 0.2, timeout=5, is_integer=False)
        self.set_param("RTL_RETURN_ALT", 3.0, timeout=5, is_integer=False)
        self.set_param("RTL_DESCEND_ALT", 1.0, timeout=5, is_integer=False)

        rospy.logwarn("waiting for topic")
        self.wait_for_topics()

        # self.set_param("MPC_XY_CRUISE", 1.0, timeout=5, is_integer=False)
        # self.set_param("MPC_VEL_MANUAL", 1.0, timeout=5, is_integer=False)
        # self.set_param("MPC_ACC_HOR", 1.0, timeout=5, is_integer=False)
        # self.set_param("MPC_JERK_AUTO", 2.0, timeout=5, is_integer=False)
        # self.set_param("MC_PITCHRATE_MAX", 100.0, timeout=5, is_integer=False)
        # self.set_param("MC_ROLLRATE_MAX", 100.0, timeout=5, is_integer=False)

        rospy.logwarn("sending offboard")
        self.start_sending_position_setpoint()

        rospy.logwarn(
            "please tell the drone to takeoff then put the drone in offboard mode"
        )
        self.wait_for_takeoff()
        self.wait_for_offborad()
        rospy.logwarn("wait for the drone go to the initial position")
        self.goto_position(0, 0, 0.5, 0)
        rospy.sleep(10)

        # tell rover and referee it can go
        self.ready_pub.publish(True)

    def rover_pos_callback(self, msg: Odometry):
        rover_pos = msg.pose.pose.position
        drone = np.array(
            [self.local_position.pose.position.x, self.local_position.pose.position.y]
        )
        rover = np.array([rover_pos.x, rover_pos.y])

        scale_xy = 0.5
        scale_z = 0.2

        direction = rover - drone  # (2, ), altitude is compute independently
        direction = direction + scale_xy * self.actions[self.action, :2]
        direction /= np.linalg.norm(direction)

        d_separation = 1.0
        altitude = 0.5 + scale_z * self.actions[self.action, 2]

        p_goal = rover - direction * d_separation

        yaw = np.arctan2(direction[1], direction[0])

        self.goto_position(
            x=p_goal[0], y=p_goal[1], z=altitude, yaw_deg=np.rad2deg(yaw)
        )

    def reset(self):
        # environment setup
        self.obs = np.zeros((3, 480, 640))
        self.rewards = 0
        self.terminated = [False]
        return self.obs, None

    def step(self, action):
        """interact with the environment, get observation and reward"""
        self.action = action
        rospy.loginfo(f"step with action: {action}")
        rospy.sleep(1)
        observation = self.obs  # (3, 480, 640)
        reward = self.rewards
        terminated = self.terminated
        rospy.loginfo(f"step done with rewards: {reward}")
        return observation, reward, terminated, False, None

    def start_train(self):
        for sample_phase in range(self.n_updates):
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically

            # reset lists that collect experiences of an episode (sample phase)
            ep_value_preds = torch.zeros(self.n_steps_per_update, device=self.device)
            ep_rewards = torch.zeros(self.n_steps_per_update, device=self.device)
            ep_action_log_probs = torch.zeros(
                self.n_steps_per_update, device=self.device
            )
            masks = torch.zeros(self.n_steps_per_update)

            # at the start of training reset all envs to get an initial state
            if sample_phase == 0:
                states, info = self.reset()

            # play n steps in our parallel environments to collect data
            for step in range(self.n_steps_per_update):
                # select an action A_{t} using S_{t} as input for the agent
                (
                    actions,
                    action_log_probs,
                    state_value_preds,
                    entropy,
                ) = self.agent.select_action(states)

                # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
                states, rewards, terminated, truncated, infos = self.step(
                    actions.cpu().numpy()
                )

                ep_value_preds[step] = torch.squeeze(state_value_preds)
                ep_rewards[step] = torch.tensor(rewards, device=self.device)
                ep_action_log_probs[step] = action_log_probs

                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                masks[step] = torch.tensor(not terminated)

                rospy.sleep(1)

            # calculate the losses for actor and critic
            critic_loss, actor_loss = self.agent.get_losses(
                ep_rewards,
                ep_action_log_probs,
                ep_value_preds,
                entropy,
                masks,
                self.gamma,
                self.lam,
                self.ent_coef,
                self.device,
            )

            # update the actor and critic networks
            self.agent.update_parameters(critic_loss, actor_loss)

            # log the losses and entropy
            self.critic_losses.append(critic_loss.detach().cpu().numpy())
            self.actor_losses.append(actor_loss.detach().cpu().numpy())
            self.entropies.append(entropy.detach().mean().cpu().numpy())
            rospy.loginfo(f"actor loss: {self.actor_losses[-1]}, critic_loss: {self.critic_losses[-1]}")
        rospy.loginfo("train end")

    def start_sending_position_setpoint(self):
        self.pos_thread.start()

    def __del__(self):
        # rospy.signal_shutdown("finished script")
        if self.pos_thread.is_alive():
            self.pos_thread.join()

    def show_result(self):
        """plot the results"""
        rospy.loginfo("showing result")
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
        fig.suptitle(
            f"Training plots for {self.agent.__class__.__name__} in the environment (n_steps_per_update={self.n_steps_per_update})"
        )

        # episode return
        axs[0][0].set_title("Episode Returns")
        episode_returns = np.array(self.return_queue)
        axs[0][0].plot(np.arange(len(episode_returns)), episode_returns)
        axs[0][0].set_xlabel("Number of episodes")

        # entropy
        axs[1][0].set_title("Entropy")
        entropy_moving_average = np.array(self.entropies)
        axs[1][0].plot(entropy_moving_average)
        axs[1][0].set_xlabel("Number of updates")

        # critic loss
        axs[0][1].set_title("Critic Loss")
        critic_losses_moving_average = np.array(self.critic_losses)
        axs[0][1].plot(critic_losses_moving_average)
        axs[0][1].set_xlabel("Number of updates")

        # actor loss
        axs[1][1].set_title("Actor Loss")
        actor_losses_moving_average = np.array(self.actor_losses)
        axs[1][1].plot(actor_losses_moving_average)
        axs[1][1].set_xlabel("Number of updates")

        plt.tight_layout()

        now = int(round(time.time() * 1000))
        now_str = time.strftime("%Y%m%d%H%M%S", time.localtime(now / 1000))

        plt.savefig(f"train-{now_str}.png")


if __name__ == "__main__":
    session = TrainSession(
        n_updates=2000, n_steps_per_update=8
    )  # 不要把train写入init中，否则不会报错
    session.setup_env()
    session.start_train()
    session.show_result()

    # waiting for thread termination
    rospy.spin()
