#!/usr/bin/env python3
import rospy
import message_filters
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool
import numpy as np
import json
import time


class Referee:
    def __init__(self):
        rospy.init_node("referee")
        self.sub_rover = rospy.Subscriber("rover", Odometry, self.rover_callback)
        self.sub_drone = rospy.Subscriber("drone", Odometry, self.drone_callback)
        self.pub_score = rospy.Publisher("score", Float32, queue_size=10)
        self.sub_drone_ready = rospy.Subscriber(
            "drone_ready", Bool, self.drone_ready_callback
        )
        self.sub_rover_finished = rospy.Subscriber(
            "rover_finished", Bool, self.rover_finished_callback
        )
        self.drone_position = None
        self.rover_position = None
        self.drone_ready = False
        self.rover_finished = False
        self.start = None
        self.sum = 0
        self.samples = 0
        self.datas = {}
        self.data_keys = ["distance", "inst_score", "sum", "samples", "score"]
        for key in self.data_keys:
            self.datas[key] = []
        rospy.spin()

    def drone_callback(self, odom):
        self.drone_position = odom.pose.pose.position

    def rover_callback(self, odom):
        """score when we see the rover, use last known drone position"""
        if not self.drone_ready:
            return

        self.rover_position = odom.pose.pose.position

        # if time expired
        if self.rover_finished:
            now = int(round(time.time() * 1000))
            time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(now / 1000))
            with open(f"~/refree{time_str}.json", "w") as f:
                json.dump(self.datas, f)
            rospy.logwarn("trial finished, final score: %f", self.score)
            rospy.signal_shutdown("finished")
            return

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
        self.sum += inst_score
        self.samples += 1
        self.score = self.sum / self.samples
        self.datas["distance"].push(distance)
        self.datas["inst_score"].push(inst_score)
        self.datas["sum"].push(sum)
        self.datas["samples"].push(self.samples)
        self.datas["score"].push(self.score)
        self.pub_score.publish(Float32(self.score))

    def drone_ready_callback(self, msg):
        self.drone_ready = msg.data

    def rover_finished_callback(self, msg):
        self.rover_finished = msg.data


if __name__ == "__main__":
    Referee()
