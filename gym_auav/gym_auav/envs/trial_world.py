#!/usr/bin/env python3
import rospy, math, roslaunch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from cv_bridge import CvBridge
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool, Header
from sensor_msgs.msg import Image, CameraInfo

# from contacts_msg.msg import Contacts


class TrialWorldEnv(gym.Env):
    metadata = {"render_modes": ["gui"]}

    def __init__(
        self,
        render_mode=None,
        launch_filename="/home/ubuntu/catkin_ws/src/auav_2022_sample/auav_2022_sample/launch/sim.launch",
        world="worlds/trial_1.world",
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        rospy.init_node("gym_env", anonymous=True)

        # Subscriber
        self.drone_ready_sub = rospy.Subscriber(
            "/drone/drone_ready", Bool, self.drone_ready_callback
        )
        self.camera_info_sub = rospy.Subscriber(
            "/drone/camera/color/camera_info", CameraInfo, self.camera_info_callback
        )
        self.camera_sub = rospy.Subscriber(
            "/drone/camera/color/image_raw", Image, self.image_callback, queue_size=1
        )
        self.rover_finished_sub = rospy.Subscriber(
            "/drone/rover_finished", Bool, self.rover_finished_callback
        )
        self.rover_sub = rospy.Subscriber("/drone/rover", Odometry, self.rover_callback)
        self.local_pos_sub = rospy.Subscriber(
            "/drone/mavros/local_position/pose",
            PoseStamped,
            self.local_position_callback,
        )
        # self.contact_sub = rospy.Subscriber(
        #     "/drone/contacts", Contacts, self.contact_callback
        # )

        # Publisher
        self.pos_setpoint_pub = rospy.Publisher(
            "/drone/mavros/setpoint_position/local", PoseStamped, queue_size=1
        )

        # RL space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )  # w,h,x,y
        self.action_space = spaces.Discrete(5)  # 动作空间，表示前进的方向
        self._action_to_direction = {
            0: np.array([1, 0, 0]),
            1: np.array([0, 1, 0]),
            2: np.array([-1, 0, 0]),
            3: np.array([0, -1, 0]),
            4: np.array([0, 0, 0]),
        }

        # parse launch args
        gui = "true" if render_mode == "gui" else "false"
        cli_args = [launch_filename, f"world:={world}", f"gui:={gui}"]
        roslaunch_args = cli_args[1:]
        self.roslaunch_file = [
            (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)
        ]

        self.bridge = CvBridge()
        self.pos = PoseStamped()
        # call reset

    def init_variable(self):
        """init private variable"""
        self.drone_ready = False
        self.rover_finished = False
        self.observation = None
        self.launch = None
        self.score = 0
        self.local_position = None
        self.camera_info = None

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.init_variable()
        self.wait_for_launch()
        info = None
        return self.observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        direction /= np.linalg.norm(direction)

        d_separation = 1.0
        p_goal = self.local_position + direction * d_separation

        yaw = np.arctan2(direction[1], direction[0])
        self.pos.pose.position.x = p_goal[0]
        self.pos.pose.position.y = p_goal[1]
        self.pos.pose.position.z = p_goal[2]
        yaw = math.radians(np.rad2deg(yaw))
        quaternion = quaternion_from_euler(0, 0, yaw)
        self.pos.pose.orientation = Quaternion(*quaternion)
        self.pos.header = Header()
        self.pos.header.frame_id = "drone"
        self.pos.header.stamp = rospy.Time.now()
        self.pos_setpoint_pub.publish(self.pos)
        rospy.loginfo(
            f"[trial_world]Target position: ({p_goal[0]:.2f}, {p_goal[1]:.2f}, {p_goal[2]:.2f})"
        )
        # An episode is done iff the agent has reached the target
        terminated = self.rover_finished
        reward = self.score
        observation = self.observation

        return observation, reward, terminated, False, None

    def wait_for_launch(self):
        """wait for launch start"""
        if self.launch != None:
            self.launch.shutdown()

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)  # 创建一个父进程并获取uuid
        roslaunch.configure_logging(uuid)

        launch = roslaunch.parent.ROSLaunchParent(uuid, self.roslaunch_file)
        launch.start()  # 以uuid为父进程，启动launch文件
        rospy.loginfo("[trail_world]Wait for drone ready.")

        # 阻塞等待准备完成
        while not self.drone_ready:
            rospy.sleep(10)

    def drone_ready_callback(self, ready):
        self.drone_ready = ready

    def camera_info_callback(self, msg: CameraInfo):
        """Callback from camera projetion"""
        self.camera_info = msg

    def image_callback(self, msg: Image):
        if self.camera_info is None:
            rospy.logerr("no camera info")
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.observation = img

    def rover_callback(self, odom):
        """score when we see the rover, use last known drone position"""
        if not self.drone_ready:
            return

        self.rover_position = odom.pose.pose.position

        # abort if no drone position
        if self.drone_position is None:
            return

        distance = np.linalg.norm(
            np.array(
                [
                    self.rover_position.x - self.drone_position.x,
                    self.rover_position.y - self.drone_position.y,
                    self.rover_position.z - self.drone_position.z,
                ]
            )
        )
        inst_score = 0
        if distance < 5:
            inst_score = 1 - np.abs(distance - 1) / 4
        self.score = inst_score

    def rover_finished_callback(self, msg):
        self.rover_finished = msg.data

    def goto_position(self, x, y, z, yaw_deg):
        """goto position"""
        # set a position setpoint
        self.pos.pose.position.x = x
        self.pos.pose.position.y = y
        self.pos.pose.position.z = z

        # For demo purposes we will lock yaw/heading to north.
        yaw = math.radians(yaw_deg)
        quaternion = quaternion_from_euler(0, 0, yaw)
        self.pos.pose.orientation = Quaternion(*quaternion)

    def local_position_callback(self, data: PoseStamped):
        self.local_position = data

    def contact_callback(self, msg):
        if msg.collision_1 != "default":
            self.score -= 100  # 如果发生碰撞，获得负奖励

    def close(self):
        if self.launch != None:
            self.launch.shutdown()
        pass
