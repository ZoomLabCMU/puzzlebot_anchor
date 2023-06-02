import rospy
import signal
import sys
import time
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TransformStamped
from puzzlebot_assembly.tcp_interface import TCPBridge
from puzzlebot_assembly.robots import Robots
from puzzlebot_assembly.utils import *

class RobotInterface:
    def __init__(self, N, H=3, dts=0.1):
        self.N = N
        self.H = H # time horizon of previous velocity
        self.pose = np.zeros([3, N])
        self.prev_pose = np.ones([3, N])
        self.prev_cmd_vel = np.zeros([2, N]) + 1e-4
        self.pose_time = np.zeros(N)
        self.cmd_vel = np.zeros([2, N]) # left and right PWM
        self.prev_cmd_vel = np.zeros([2, N]) + 1e-4
        self.prev_vel = np.zeros([2, N, 2*H])
        self.prev_inst_vel = np.zeros([2, N, H])
        self.curr_vel = np.zeros([2, N])
        self.logger = []
        self.dts = dts

        self.control_param = {'Pv': 0.5, 'Pw': 0.02, 'Iv': 0.3, 'Iw': 0.01, 
                        'opposite_scale_v': 10, 'opposite_scale_w': 10}
        self.robot_limit = {'vmax': 0.2, 'wmax': 6.0}
    
    def fit_pid(self, gdu, th=1e-3, v_min=0, v_max=250, same_sign=True, 
                pilot_ids=[]):
        """
        gdu: 2*N vec
        return: 2-by-N motor PWM signal
        """
        N = self.N
        cparam = self.control_param
        vel = self.prev_cmd_vel.copy()
        print('gdu:', gdu)

        zero_mask = np.sum(np.abs(gdu), axis=0) < th
        if np.all(zero_mask): 
            return np.zeros([2, N])
        
        Pv, Pw = cparam['Pv'], cparam['Pw']
        Iv, Iw =  cparam['Iv'], cparam['Iw']
        ov, ow = cparam['opposite_scale_v'], cparam['opposite_scale_w']
        curr_vel = self.curr_vel
        pids = np.zeros([4, N]) + np.array([[Pv, Pw, Iv, Iw]]).T

        err = gdu - curr_vel
        intg_err = gdu - np.average(self.prev_vel[:, :, :-1], axis=2)
        is_opposite = ((np.sign(gdu) * np.sign(curr_vel)) < 0)
        is_opposite[0, :] &= (np.abs(gdu[0, :] - curr_vel[0, :]) > 1e-2)
        is_opposite[1, :] &= (np.abs(gdu[1, :] - curr_vel[1, :]) > 1e-1)
        pids[0, is_opposite[0, :]] *= ov
        pids[1, is_opposite[1, :]] *= ow
        dv = (Pv * err + Iv * intg_err)[0, :]
        dw = (Pw * err + Iw * intg_err)[1, :]

        # pilot robot fiters
        dw[pilot_ids] = 0
        
        dR = (dv / 0.0007 + dw / 0.00267) / 2
        dL = (dv / 0.0007 - dw / 0.00267) / 2
        
        vel[0, :] += dL
        vel[1, :] += dR

        # clip the PWM values to be the same sign
        if same_sign:
            gdu_sign = np.sign(gdu[0, :])
            vel[0, (np.sign(vel[0, :]) * gdu_sign) < 0] = 1
            vel[1, (np.sign(vel[1, :]) * gdu_sign) < 0] = 1

        # normalize vels to the range limit
        vel_max = np.max(np.abs(vel), axis=0)
        vel_max_idx = vel_max > v_max
        vel[:, vel_max_idx] = np.divide(vel*v_max, vel_max)[:, vel_max_idx]
        pilot_max = 150 if np.all(self.pose[0, pilot_ids] < -0.083) else 250
        vel[:, pilot_ids] = np.clip(vel[:, pilot_ids], 
                                    -pilot_max, pilot_max)

        vel[:, zero_mask] = 0

        self.cmd_vel = vel.astype(int)
        self.prev_cmd_vel[:, :] = self.cmd_vel[:, :]
        return self.cmd_vel

    def update_vels(self):
        N = self.N
        vmax, wmax = self.robot_limit['vmax'], self.robot_limit['wmax']
        curr = (self.pose_time.copy(), self.pose.copy())
        if len(self.logger) < 1:
            self.logger.append(curr)
            return
        prev = self.logger[-1]
        self.logger.append(curr)

        t_diff = curr[0] - prev[0]
        curr_inst_vel = np.zeros([2, N])
        for i in range(N):
            t = t_diff[i] + 1e-7
            pr = get_R(prev[1][2, i])
            x_diff = pr.T.dot(curr[1][0:2, i]) - pr.T.dot(prev[1][0:2, i])
            v = x_diff[0] / t
            w = (curr[1][2, i] - prev[1][2, i]) / t
            if v > vmax or v < -vmax: return
            if w > wmax or w < -wmax: return
            if np.sum(np.abs([v, w])) < 1e-6: return
            curr_inst_vel[:, i] = [v, w]

        self.prev_inst_vel[:, :, :-1] = self.prev_inst_vel[:, :, 1:]
        self.prev_inst_vel[:, :, -1] = curr_inst_vel[:, :]
        self.curr_vel = np.average(self.prev_inst_vel, axis=2)
        self.prev_vel[:, :, :-1] = self.prev_vel[:, :, 1:]
        self.prev_vel[:, :, -1] = self.curr_vel[:, :]

class HardwareWrap:
    def __init__(self, N):
        self.N = N
        self.robot_int = RobotInterface(N)
        self.tcp_com = None
        self.use_vicon = True

        rospy.init_node('hardware_control', anonymous=True, 
                    log_level=rospy.DEBUG) 
        self.sub = [None] * N
        self.sub_u = None
        self.pub_x = None
        self.pilot_ids = []
        
        self.u = np.zeros([2, N])
        
        rospy.loginfo('Robot controller initialized.')

    def init_pub_sub(self, ips):
        # vicon callback on/off
        if self.use_vicon:
            for i in range(self.N):
                ip = ips[i]
                self.sub[i] = rospy.Subscriber('vicon/p%d/p%d' % (ip, ip), TransformStamped, self.vicon_cb, i)
                if ip == 206:
                    self.pilot_ids.append(i)

        self.sub_u = rospy.Subscriber('vel_array', 
                            Float32MultiArray, self.update_u)
        self.pub_x = rospy.Publisher('pose_array',
                            Float32MultiArray, queue_size=1)

    def update_u(self, data):
        self.u = np.array(data.data).reshape([self.N, 2]).T

    def setup(self):
        N = self.N
        tcp_com = TCPBridge(N)
        tcp_com.start_listen()
        self.init_pub_sub(tcp_com.robot_ips)
        self.tcp_com = tcp_com
        rospy.on_shutdown(tcp_com.end)

    def vicon_cb(self, data, i):
        if self.robot_int is None:
            return

        pose = np.zeros(3)
        pose[0] = data.transform.translation.x
        pose[1] = data.transform.translation.y
        q =  data.transform.rotation
        quat = (q.x, q.y, q.z, q.w)
        pose[2] = yaw_from_quaternion(quat)

        self.robot_int.prev_pose[:, i] = self.robot_int.pose[:, i]
        self.robot_int.pose[:, i] = pose
        self.robot_int.pose_time[i] = time.time()

    #  def signal_handler(self, sig, frame):
        #  print('Interupt with Ctrl+C!')

    def start(self):
        N = self.N
        tcp_com = self.tcp_com
        r = self.robot_int
        freq = 50
        rate = rospy.Rate(freq)

        state = 0
        pose_stable = False
        pub_pose = Float32MultiArray()

        while not rospy.is_shutdown():
            rate.sleep()

            if self.use_vicon and (np.count_nonzero(r.pose) < 3*N):
                continue

            if (not pose_stable) and np.all((r.prev_pose - r.pose) < 1e-4):
                pose_stable = True
                rospy.loginfo('Pose is stable')

            if self.use_vicon and (not pose_stable): continue

            pub_pose.data = r.pose.T.flatten().tolist()
            self.pub_x.publish(pub_pose)
            r.update_vels()
            r.dts = tcp_com.dts

            vel = r.fit_pid(self.u, same_sign=True, pilot_ids=self.pilot_ids)
            if vel is None:
                continue
            
            tcp_com.send(vel)

        tcp_com.end()

