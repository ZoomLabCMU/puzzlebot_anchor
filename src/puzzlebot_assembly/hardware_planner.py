import rospy
import time
import numpy as np
from std_msgs.msg import Float32MultiArray
from puzzlebot_assembly.control import Controller
from puzzlebot_assembly.robots import Robots
from puzzlebot_assembly.utils import *

class HardwarePlanner:
    def __init__(self, N, c_param, controller, rsys, pilot_ids=[]):
        self.N = N
        self.c_param = c_param
        self.ctl = controller
        self.rsys = rsys
        self.pilot_ids = pilot_ids
        
        rospy.init_node('hardware_planner', anonymous=True, 
                    log_level=rospy.DEBUG)

        self.u_pub = rospy.Publisher('vel_array', 
                                Float32MultiArray, 
                                queue_size=1)
        self.x_sub = rospy.Subscriber('pose_array', 
                                Float32MultiArray, 
                                self.update_x)
        self.x = None
        self.u = None

        rospy.loginfo('Robot planner initialized.')

    def update_x(self, data):
        self.x = np.array(data.data)
    
    def start(self):
        N = self.N
        rsys = self.rsys
        freq = 10
        rate = rospy.Rate(freq)
        have_pose = False
        pub_vel = Float32MultiArray()

        while not rospy.is_shutdown():
            rate.sleep()

            if (self.x is not None) and (not have_pose):
                rsys.setup(self.x.reshape([N, 3]).T)
                rospy.loginfo("Planner setup with %s\n" % str(self.x))
                have_pose = True
            if not have_pose: continue
            
            rsys.x = self.x
            is_done = rsys.step(rsys.x, rsys.prev_u, time.time())
            self.u = rsys.u
            pub_vel.data = self.u.tolist()
            self.u_pub.publish(pub_vel)
