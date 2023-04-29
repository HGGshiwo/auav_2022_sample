#!/usr/bin/env python3
import torch
from A2C import A2C
import matplotlib.pyplot as plt
import numpy as np
import rospy
import matplotlib.pyplot as plt
import numpy as np
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from mavros_msgs.msg import State
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Quaternion, PointStamped, Point
import math

class Train:
    def __init__(
        self,
        n_envs=1,
        n_updates=1000,
        n_steps_per_update=128,
        randomize_domain=False,
        gamma=0.999,
        lam=0.95,  # hyperparameter for GAE
        ent_coef=0.01,  # coefficient for the entropy bonus (to encourage exploration)
        actor_lr=0.001,
        critic_lr=0.005,
        use_cuda=False,
    ) -> None:
        rospy.init_node("train")

        # environment hyperparams
        self.n_envs = n_envs
        self.n_updates = n_updates
        self.n_steps_per_update = n_steps_per_update
        self.randomize_domain = randomize_domain
        self.rover_pos = PointStamped()
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

        # init the agent
        obs_shape = self.observation.shape
        action_shape = len(self.actions)
        self.agent = A2C(
            obs_shape, action_shape, self.device, critic_lr, actor_lr, n_envs
        )

        self.critic_losses = []
        self.actor_losses = []
        self.entropies = []
        self.return_queue = []
        self.obs = None
        self.rewards = None
        self.terminated = None
        self.ready = False
        self.bridge = CvBridge()
        self.actions = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, -1],
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [-1, 0, 0],
            ]
        )

        self.finished_sub = rospy.Subscriber(
            "/rover/finished", Bool, self.finished_callback
        )
        self.reward_sub = rospy.Subscriber("/inst_score", Float32, self.reward_callback)
        self.ready_sub = rospy.Subscriber("/drone/ready", Bool, self.ready_callback)
        self.fly_towards_pub = rospy.Publisher("/fly_towards", Point, queue_size=10)
        self.rov_pos_sub = rospy.Subscriber('rover/point',
                                            PointStamped,
                                            self.rover_pos_callback)
        self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose',
                                              PoseStamped,
                                              self.local_position_callback)
        rospy.loginfo("wait for drone ready")
        while not self.ready:
            rospy.sleep(10)
        rospy.loginfo("start train")
        self.run()

    def local_position_callback(self, data: PoseStamped):
        self.local_position = data

    def rover_pos_callback(self, msg: PointStamped):
        self.rover_pos = msg
        euler_current = euler_from_quaternion([
            self.local_position.pose.orientation.x,
            self.local_position.pose.orientation.y,
            self.local_position.pose.orientation.z,
            self.local_position.pose.orientation.w], axes='rzyx')

        drone = np.array([
            self.local_position.pose.position.x,
            self.local_position.pose.position.y])

        rover = np.array([
            self.rover_pos.point.x,
            self.rover_pos.point.y])

        direction = rover - drone
        direction /= np.linalg.norm(direction)

        d_separation = 1.0
        altitude = 0.5
        
        p_goal = rover - direction*d_separation

        yaw = np.arctan2(direction[1], direction[0])
      
        self.goto_position(x=p_goal[0], y=p_goal[1], z=altitude, yaw_deg=np.rad2deg(yaw))


    def goto_position(self, x, y, z, yaw_deg):
        """goto position"""
        # set a position setpoint
        self.pos.pose.position.x = x
        self.pos.pose.position.y = y
        self.pos.pose.position.z = z

        euler_current = euler_from_quaternion(
                [self.local_position.pose.orientation.x,
                self.local_position.pose.orientation.y,
                self.local_position.pose.orientation.z,
                self.local_position.pose.orientation.w], axes='rzyx')

        # For demo purposes we will lock yaw/heading to north.
        yaw = math.radians(yaw_deg)
        quaternion = quaternion_from_euler(0, 0, yaw)
        self.pos.pose.orientation = Quaternion(*quaternion)
    
    def finished_callback(self, msg):
        self.terminated = msg.data

    def reward_callback(self, msg):
        self.reward = msg.data

    def camera_info_callback(self, msg: CameraInfo):
        """Callback from camera projetion"""
        self.camera_info = msg

    def image_callback(self, msg):
        if self.camera_info is None:
            rospy.logerr("no camera info")
            return
        obs = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.obs = np.array([obs])  # 480*640*3

    def ready_callback(self, msg):
        self.ready = msg.data

    def reset(self):
        # environment setup
        self.obs = np.zeros((1, 3, 480, 640))
        self.rewards = np.zeros((1, 1))
        self.terminated = False

    def step(self, action):
        """interact with the environment, get observation and reward"""
        direction = self.actions[action[0]]

        # fly towards the given direction
        self.pos.pose.position.x += direction[0]
        self.pos.pose.position.y += direction[1]
        self.pos.pose.position.z += direction[2]

        yaw = np.arctan2(direction[1], direction[0])
        quaternion = quaternion_from_euler(0, 0, yaw)
        self.pos.pose.orientation = Quaternion(*quaternion)

        observation = self.observation  # 1*480*640*3
        reward = np.array([self.reward])
        terminated = [self.terminated]
        return observation, reward, terminated, False, None

    def run(self):
        for sample_phase in range(self.n_updates):
            # we don't have to reset the envs, they just continue playing
            # until the episode is over and then reset automatically

            # reset lists that collect experiences of an episode (sample phase)
            ep_value_preds = torch.zeros(
                self.n_steps_per_update, self.n_envs, device=self.device
            )
            ep_rewards = torch.zeros(
                self.n_steps_per_update, self.n_envs, device=self.device
            )
            ep_action_log_probs = torch.zeros(
                self.n_steps_per_update, self.n_envs, device=self.device
            )
            masks = torch.zeros(
                self.n_steps_per_update, self.n_envs, device=self.device
            )

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
                states, rewards, terminated, truncated, infos = step(
                    actions.cpu().numpy()
                )

                ep_value_preds[step] = torch.squeeze(state_value_preds)
                ep_rewards[step] = torch.tensor(rewards, device=self.device)
                ep_action_log_probs[step] = action_log_probs

                # add a mask (for the return calculation later);
                # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
                masks[step] = torch.tensor([not term for term in terminated])

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
            self.return_queue.append(np.mean(ep_rewards))

    def plot(self):
        """plot the results"""

        rolling_length = 20
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
        fig.suptitle(
            f"Training plots for {self.agent.__class__.__name__} in the LunarLander-v2 environment \n \
                    (n_envs={self.n_envs}, n_steps_per_update={self.n_steps_per_update}, randomize_domain={self.randomize_domain})"
        )

        # episode return
        axs[0][0].set_title("Episode Returns")
        episode_returns_moving_average = (
            np.convolve(
                np.array(self.return_queue).flatten(),
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
        )
        axs[0][0].plot(
            np.arange(len(episode_returns_moving_average)) / self.n_envs,
            episode_returns_moving_average,
        )
        axs[0][0].set_xlabel("Number of episodes")

        # entropy
        axs[1][0].set_title("Entropy")
        entropy_moving_average = (
            np.convolve(np.array(self.entropies), np.ones(rolling_length), mode="valid")
            / rolling_length
        )
        axs[1][0].plot(entropy_moving_average)
        axs[1][0].set_xlabel("Number of updates")

        # critic loss
        axs[0][1].set_title("Critic Loss")
        critic_losses_moving_average = (
            np.convolve(
                np.array(self.critic_losses).flatten(),
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
        )
        axs[0][1].plot(critic_losses_moving_average)
        axs[0][1].set_xlabel("Number of updates")

        # actor loss
        axs[1][1].set_title("Actor Loss")
        actor_losses_moving_average = (
            np.convolve(
                np.array(self.actor_losses).flatten(),
                np.ones(rolling_length),
                mode="valid",
            )
            / rolling_length
        )
        axs[1][1].plot(actor_losses_moving_average)
        axs[1][1].set_xlabel("Number of updates")

        plt.tight_layout()
        plt.savefig("train.png")


if __name__ == "__main__":
    Train()
