#!/usr/bin/env python

import sys
import rospy
from puzzlebot_assembly.hardware_wrap import HardwareWrap 
from puzzlebot_assembly.robots import Robots, RobotParam
from puzzlebot_assembly.control import Controller, ControlParam
from puzzlebot_assembly.behavior_lib import BehaviorLib

if __name__ == "__main__":
    try:
        N = rospy.get_param("/robot_num")
        hc = HardwareWrap(N)
        hc.setup()
        hc.start()
    except rospy.ROSInterruptException:
        pass
