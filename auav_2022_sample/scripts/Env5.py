#!/usr/bin/env python3

# 进行一些数据采集
from offboard import MavrosOffboardPosctl
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3Stamped, TwistStamped
from mavros_msgs.msg import ExtendedState, State, Thrust
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
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from filterpy.kalman import KalmanFilter
import time
import json


class Env(MavrosOffboardPosctl):
    def __init__(
        self,
        use_odom=True,
        use_KF=False,
        use_RL=True,
        verbose=False,
        state_mode="pos",
        action_mode="vel",
    ):
        """init for Env
        use_odom:   use true position of rover if True
        use_KF:     use kalman filter for rover position estimation
        use_RL:     use rl output
        verbose:    output first episode data if True
        state_mode:   choose state space, use position if "pos", use image if "img"
        action_mode:    choose action space, use velocity control if "vel", use position control if "pos"
        """
        rospy.init_node("offboard_node")
        self.use_odom = use_odom  # whether to use real position of rover
        self.rewards = 0
        self.terminated = False
        self.score = 0
        self.env_ready = False
        
        self.use_KF = use_KF
        self.kf = KalmanFilter(dim_x=2, dim_z=2)
        self.verbose = verbose
        self.action_mode = action_mode
        self.state_mode = state_mode

        self.actions = None
        if self.action_mode == "vel":
            actions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            self.actions = np.array(actions, dtype=np.float64)
        else:
            actions = [[0, 0]]
            for i in range(2, 7):
                actions.extend([[-0.2 * i, 0], [0.2 * i, 0], [0, 0.2 * i], [0, -0.2 * i]])
            self.actions = np.array(actions, dtype=np.float64)

        self.action_shape = len(self.actions)
        self.obs_shape = None # 为了读取shape
        if self.state_mode == "img":
            self.obs_shape = (3, 480, 640)
        else:
            self.obs_shape = 10
        self.return_queue = []  # 记录下每个episode的return
        self.bridge = CvBridge()
        self.extended_state = ExtendedState()
        self.imu_data = Imu()

        self.drone_pos = np.zeros((2,))  # 无人机的位置
        self.rover_pos = np.zeros((2,))  # 小车的位置
        self.state = State()  # 无人机的状态
        self.pos = PoseStamped()  # 无人机目标点的位置
        self.vel = TwistStamped()
        self.radius = 0.1
        self.height = 0
        self.use_RL = use_RL

        self.rover_poses = []  # 记录一个episode中小车的位置
        self.drone_poses = []  # 记录一个episode中飞机的位置
        self.distances = []  # 记录一个episode中两者的距离
        self.drone_vels = []  # 记录无人机的速度变化图
        self.first_episode = True  # 只采集第一个episode中的数据

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
        pose_type = Odometry if use_odom else PointStamped
        pose_name = "/qualisys/rover/odom" if use_odom else "rover/point"
        self.rov_pos_sub = rospy.Subscriber(
            pose_name, pose_type, self.rover_pos_callback
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
        self.vel_sub = rospy.Subscriber(
            "mavros/local_position/velocity", TwistStamped, self.vel_callback
        )
        self.ready_pub = rospy.Publisher("ready", Bool, queue_size=10)
        self.pos_setpoint_pub = rospy.Publisher(
            "mavros/setpoint_position/local", PoseStamped, queue_size=1
        )
        self.setpoint_vel_pub = rospy.Publisher(
            "mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1
        )
        self.setpoint_thrust_pub = rospy.Publisher(
            "mavros/setpoint_attitude/thrust", Thrust, queue_size=1
        )

        # send setpoints in seperate thread to better prevent failsafe
        self.pos_thread = Thread(target=self.send_pos, args=())
        self.setup_env()
        self.env_ready = True

    def send_pos(self):
        rate = rospy.Rate(10)  # Hz

        self.pos.header = Header()
        self.pos.header.frame_id = "drone"
        self.vel.header = Header()
        self.pos.header.frame_id = "drone"

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()
            self.vel.header.stamp = rospy.Time.now()
            if self.action_mode == "vel":
                if np.abs(self.height - 0.5) > 0.1:
                    self.pos_setpoint_pub.publish(self.pos)
                else:
                    self.setpoint_vel_pub.publish(self.vel)
            else:
                self.pos_setpoint_pub.publish(self.pos)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def camera_info_callback(self, msg: CameraInfo):
        """Callback from camera projetion"""
        self.camera_info = msg

    def do_action(self, action): 
        direction = self.rover_pos - self.drone_pos
        self.distances.append(np.linalg.norm(direction).item())

        if self.action_mode == "vel":
            a_goal = self.actions[action] if self.use_RL else 0.6
            d_separation = 1
            altitude = 0.5
            norm_direction = direction / np.linalg.norm(direction)
            p_goal = self.rover_pos - norm_direction * d_separation

            yaw = np.arctan2(direction[1], direction[0])
            self.goto_position(p_goal[0], p_goal[1], altitude, np.rad2deg(yaw))

            if np.linalg.norm(direction) < 1:
                self.vel.twist.linear.x = 0
                self.vel.twist.linear.y = 0
                self.vel.twist.linear.z = 0
            else:
                self.vel.twist.linear.x = a_goal * norm_direction[0]
                self.vel.twist.linear.y = a_goal * norm_direction[1]
                self.vel.twist.linear.z = 0
        else:
            # 使用位置进行控制
            cor_direction = direction + self.actions[action] # direction after correct
            cor_direction /= np.linalg.norm(cor_direction)

            d_separation = 1.0
            altitude = 0.5

            p_goal = self.rover_pos - cor_direction * d_separation

            yaw = np.arctan2(cor_direction[1],cor_direction[0])

            self.goto_position(
                x=p_goal[0], y=p_goal[1], z=altitude, yaw_deg=np.rad2deg(yaw)
            )

    def image_callback(self, msg):
        if self.camera_info is None:
            rospy.logerr("no camera info")
            return

        if self.state_mode != "img":
            return
        
        obs = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        obs = np.array(obs).transpose((2, 0, 1))  # 3*480*640

        action = self.onState(obs)
        self.do_action(action)

    def reward_callback(self, msg):
        self.rewards = msg.data

    def return_callback(self, msg):
        self.score = msg.data

    def dump_data(self):
        """save data for paper fig"""
        now = int(round(time.time() * 1000))
        now_str = time.strftime("%Y%m%d%H%M%S", time.localtime(now / 1000))
        with open(f"{now_str}.json", "w") as f:
            data = {}
            data["drone_vels"] = self.drone_vels
            data["rover_poses"] = self.rover_poses
            data["drone_poses"] = self.drone_poses
            data["distances"] = self.distances
            json.dump(data, f)

    def finished_callback(self, msg):
        self.terminated = msg.data
        if self.terminated:
            self.return_queue.append(self.score)
            self.reset()  # auto reset
            if self.first_episode and self.verbose:
                self.dump_data()
                self.first_episode = False

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

        rospy.logwarn("sending offboard")
        self.start_sending_position_setpoint()

        rospy.logwarn(
            "please tell the drone to takeoff then put the drone in offboard mode"
        )
        self.wait_for_takeoff()
        self.wait_for_offborad()
        rospy.logwarn("wait for the drone go to the initial position")
        self.goto_position(0, 0, 0.5, 0)

    def rover_pos_callback(self, msg):
        rover_pos = msg.pose.pose.position if self.use_odom else msg.point
        self.rover_pos = np.array([rover_pos.x, rover_pos.y])
        self.rover_poses.append([rover_pos.x, rover_pos.y])
        
        if self.state_mode != "pos":
            return
        
        if self.use_KF:
            self.kf.predict()
            self.rover_pos = self.kf.x
            self.kf.update(self.rover_pos)

        direction = self.rover_pos - self.drone_pos

        obs = np.zeros((10,))
        obs[0] = direction[0]
        obs[1] = direction[1]
        obs[2] = self.imu_data.orientation.x
        obs[3] = self.imu_data.orientation.y
        obs[4] = self.imu_data.orientation.z
        obs[5] = self.imu_data.orientation.w
        obs[6] = self.imu_data.angular_velocity.x
        obs[7] = self.imu_data.angular_velocity.y
        obs[8] = self.imu_data.linear_acceleration.x
        obs[9] = self.imu_data.linear_acceleration.y

        action = self.onState(obs)
        self.do_action(action)

    def local_position_callback(self, data: PoseStamped):
        if not self.sub_topics_ready["local_pos"]:
            self.sub_topics_ready["local_pos"] = True

        self.drone_pos = np.array([data.pose.position.x, data.pose.position.y])
        self.height = data.pose.position.z
        self.drone_poses.append([data.pose.position.x, data.pose.position.y])

    def vel_callback(self, msg):
        vel = [msg.twist.linear.x, msg.twist.linear.y]
        self.drone_vels.append(vel)

    def reset(self):
        # self.rewards = 0
        self.terminated = False
        # self.score = 0

    def start_sending_position_setpoint(self):
        self.pos_thread.start()
        # self.acc_thread.start()

    def log(self, any):
        rospy.loginfo(any)

    def spin(self):
        rospy.spin()

    def exit(self):
        rospy.signal_shutdown("finished script")

    def sleep(self, t):
        rospy.sleep(t)

    def __del__(self):
        # rospy.signal_shutdown("finished script")
        if self.pos_thread.is_alive():
            self.pos_thread.join()
        if self.acc_thread.is_alive():
            self.acc_thread.join()
