#!/usr/bin/env python3
# 自动起飞: https://docs.px4.io/main/en/ros/mavros_offboard_python.html
from offboard import MavrosOffboardPosctl
import rospy
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from mavros_msgs.msg import ExtendedState, State
from mavros_msgs.srv import ParamGet, ParamSet, CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from sensor_msgs.msg import Imu
from pymavlink import mavutil
from std_msgs.msg import Bool
from threading import Thread
import numpy as np

class Auto(MavrosOffboardPosctl):

    def __init__(self):
        rospy.init_node('offboard_node')
        self.extended_state = ExtendedState()
        self.imu_data = Imu()
        self.local_position = PoseStamped()
        self.state = State()
        self.rover_pos = PointStamped()

        self.sub_topics_ready = {
            key: False
            for key in [
                'ext_state', 'local_pos', 'state', 'imu',
            ]
        }

        # ROS services
        service_timeout = 60
        rospy.loginfo("waiting for ROS services")
        try:
            rospy.loginfo("waiting for param get")
            rospy.wait_for_service('mavros/param/get', service_timeout)
            rospy.loginfo("waiting for cmd arming")
            rospy.wait_for_service("mavros/cmd/arming", service_timeout)    
            rospy.loginfo("waiting for set mode")
            rospy.wait_for_service("mavros/set_mode", service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException as e:
            rospy.logerr("failed to connect to services")

        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)    
        self.get_param_srv = rospy.ServiceProxy('mavros/param/get', ParamGet)
        self.set_param_srv = rospy.ServiceProxy('mavros/param/set', ParamSet)

        # ROS subscribers
        # self.rov_pos_sub = rospy.Subscriber('rover/point',
        #                                     PointStamped,
        #                                     self.rover_pos_callback)
        self.ext_state_sub = rospy.Subscriber('mavros/extended_state',
                                              ExtendedState,
                                              self.extended_state_callback)
        self.imu_data_sub = rospy.Subscriber('mavros/imu/data',
                                               Imu,
                                               self.imu_data_callback)
        self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose',
                                              PoseStamped,
                                              self.local_position_callback)
        self.state_sub = rospy.Subscriber('mavros/state', State,
                                          self.state_callback)
        self.fly_towards_sub = rospy.Subscriber('/fly_towards', Point, self.fly_towards) 
        
        
        self.ready_pub = rospy.Publisher('ready', Bool, queue_size=10)

        self.pos = PoseStamped()
        self.radius = 0.1

        self.pos_setpoint_pub = rospy.Publisher(
            'mavros/setpoint_position/local', PoseStamped, queue_size=1)

        # send setpoints in seperate thread to better prevent failsafe
        self.pos_thread = Thread(target=self.send_pos, args=())
        self.run()

    def fly_towards(self, msg: Point):
        """fly towards the given direction"""
        if direction.all() == None:
            direction = [msg.x, msg.y, msg.z]
        yaw = np.arctan2(direction[0], direction[1])
        yaw_deg = np.rad2deg(yaw)
        drone = np.array(
            [
                self.local_position.pose.position.x,
                self.local_position.pose.position.y,
                self.local_position.pose.position.z,
            ]
        )
        p_goal = drone + direction * 0.1
        if p_goal[2] < 0:
            rospy.logerr(f"illegal taregt: ({p_goal[0]}, {p_goal[1]}, {p_goal[2]})")
            return
        self.goto_position(p_goal[0], p_goal[1], p_goal[2], yaw_deg)

    def wait_for_offborad(self):
        last_req = rospy.Time.now()
        rate = rospy.Rate(10) # 10 Hz
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        while not rospy.is_shutdown():
            if self.state.mode != "OFFBOARD":
                if (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    rospy.loginfo_throttle(10, "waiting for offborad") 
                    continue
                if(self.set_mode_client.call(offb_set_mode).mode_sent == True):
                    rospy.loginfo("offboard mode confirmed")
                    break
                else:
                    rospy.loginfo("set mode failed, retry")
            
            rate.sleep()
            last_req = rospy.Time.now()
    
    def wait_for_takeoff(self):
        last_req = rospy.Time.now()
        rate = rospy.Rate(10) # 10 Hz            
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        while not rospy.is_shutdown(): 
            if not self.state.armed:
                if (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    rospy.loginfo_throttle(10, "waiting for takeoff")
                    continue
                if(self.arming_client.call(arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")
                    break
                else:
                    rospy.loginfo("arm call failed, retry")
            else:
                rospy.loginfo("Vehicle armed")

            last_req = rospy.Time.now()
            rate.sleep()

    def run(self):
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

        rospy.logwarn("please tell the drone to takeoff then put the drone in offboard mode")        
        self.wait_for_takeoff()
        self.wait_for_offborad()
        p_goal = [0, 0, 0.5]
        self.goto_position(x=p_goal[0], y=p_goal[1], z=p_goal[2], yaw_deg=np.rad2deg(0))
        rospy.logwarn("wait for the drone go to the initial position")
        while not self.is_at_position(x=p_goal[0], y=p_goal[1], z=p_goal[2], offset=0.1):
            rospy.sleep(10)
        
        # tell rover and referee it can go
        self.ready_pub.publish(True)

        # waiti for thread termination
        rospy.spin()


if __name__ == '__main__':
    try:
        Auto()
    except rospy.exceptions.ROSInterruptException:
        pass
