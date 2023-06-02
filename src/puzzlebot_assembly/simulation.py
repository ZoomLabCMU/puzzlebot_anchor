import math
import time
import numpy as np
import pybullet as p
from puzzlebot_assembly.utils import *
from puzzlebot_assembly.robots import Robots
from puzzlebot_assembly.control import Controller, ControlParam
from puzzlebot_assembly.behavior_lib import BehaviorLib

class SimUtils:
    def get_lr(self, v, w, L=0.04, R=0.01):
        vl = ((2 * v) - (w * L)) / (2 * R)
        vr = ((2 * v) + (w * L)) / (2 * R)
        return vl, vr

    def get_anchor_vel(self, s, flip=1, 
                    kp=1.0, kn=1.0, vmax=2.0, eth=1e-2):
        '''
        flip: 1 for left, -1 for right
        '''
        if np.abs(s / np.pi) < eth: return 0
        vel = 0
        if (s * flip) > 0:
            vel = - kp * s
        else:
            vel = - kn * s
        vel = np.clip(vel, -vmax, vmax)
        return vel

    def get_anchor_force(self, s, kp=1e0, kn=5, 
                            max_p=0.6, max_n=1.5, 
                            eth=1e-2):
        if np.abs(s) < eth: return 0
        f = 0
        if s > 0:
            f = kp * s
            f = np.min([max_p, f])
        else:
            f = - kn * s
            f = np.min([max_n, f])
        return f

    def set_vel(self, body, joint, v, f):
        p.setJointMotorControl2(body, joint, 
                        controlMode=p.VELOCITY_CONTROL, 
                        targetVelocity=v, 
                        force=f)

class BulletSim:
    def __init__(self, N, c_param, controller, robot_system,
                    is_anchor_separate=True):
        self.N = N
        self.urdfs = {}

        # robot and anchor simulation handles
        self.ground_handle = 0
        self.rh = [] 
        self.ah = []
        self.is_anchor_separate = is_anchor_separate

        self.start_pos = []
        self.joint_id = {}

        self.c_param = c_param
        self.ctl = controller
        self.rsys = robot_system
        self.sim_utils = SimUtils()

    def setup(self, start):
        assert(self.N == start.shape[1])
        self.start_pos = start

        # initialize environment
        cid = p.connect(p.GUI) # or DIRECT for no gui
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(1)

        # set up visualizer 
        p.resetDebugVisualizerCamera(0.1, 0, -85, [0.08,0.08,0.3])

        # set up robots
        self.rsys.setup(start=start)

    def load_urdf(self, robot_file, anchor_file, env_file):
        self.urdfs['robot'] = robot_file
        self.urdfs['anchor'] = anchor_file
        self.urdfs['env'] = env_file

        self.ground_handle = p.loadURDF(self.urdfs['env'])

        start = self.start_pos
        assert(len(start) > 0, "Please call setup first!")
        for i in range(self.N):
            r = p.loadURDF(
                    "urdf/puzzlebot.urdf",
                    [start[0, i], start[1, i], 0.005], 
                    [0, 0, 0, 1]
                    )
            self.rh.append(r)
            if self.is_anchor_separate:
                ac = p.loadURDF(
                        "urdf/puz_anchor.urdf",
                        [start[0, i]-0.024, start[1, i], 0.032], # hard coded for now
                        p.getQuaternionFromEuler([0, 0, np.pi])
                        )
                self.ah.append(ac)

        self.process_urdf() # get the joint id handles
        self.init_dynamics() 

    def process_urdf(self):
        joint_id = {'left': -1, 'right': -1, 
                    'front': -1, 'back': -1,
                    'left_anchor': -1, 'right_anchor': -1,
                    'battery': -1
                    }
        r = self.rh[0]
        for jid in range(p.getNumJoints(r)):
            info = p.getJointInfo(r, jid)
            if info[1].decode("utf-8") == "left_wheel_joint":
                joint_id['left'] = jid
            elif info[1].decode("utf-8") == "right_wheel_joint":
                joint_id['right'] = jid
            elif info[1].decode("utf-8") == "front_wheel_joint":
                joint_id['front'] = jid
            elif info[1].decode("utf-8") == "back_wheel_joint":
                joint_id['back'] = jid
            elif info[1].decode("utf-8") == "battery_joint":
                joint_id['battery'] = jid

        ac = r
        if self.is_anchor_separate:
            ac = self.ah[0]
        for jid in range(p.getNumJoints(ac)):
            info = p.getJointInfo(ac, jid)
            if "anchor_left_joint" in info[1].decode("utf-8"):
                joint_id['left_anchor'] = jid
            elif "anchor_right_joint" in info[1].decode("utf-8"):
                joint_id['right_anchor'] = jid
        assert(-1 not in joint_id.values())
        
        self.joint_id = joint_id

    def init_dynamics(self):
        p.changeDynamics(self.ground_handle, 0, lateralFriction=1e8)
        ac = self.ah[0] if self.is_anchor_separate else self.rh[0]
        anchor_angle = 0.3
        max_anchor_vel = 2.0
        for i in range(self.N):
            r = self.rh[i]
            # make the ball bearings frictionless
            p.changeDynamics(r, self.joint_id['front'], lateralFriction=0)
            p.changeDynamics(r, self.joint_id['back'], lateralFriction=0)
            p.changeDynamics(r, self.joint_id['left'], 
                        lateralFriction=2.0)
            p.changeDynamics(r, self.joint_id['right'], 
                        lateralFriction=2.0)
            p.changeDynamics(r, self.joint_id['battery'], lateralFriction=0)

            ac = r
            if self.is_anchor_separate:
                ac = self.ah[i]
            p.changeDynamics(ac, self.joint_id['left_anchor'], 
                    lateralFriction=0.1, 
                    jointLowerLimit=(- anchor_angle), 
                    jointUpperLimit=(np.pi/2 - anchor_angle),
                    maxJointVelocity=max_anchor_vel)
            p.changeDynamics(ac, self.joint_id['right_anchor'], 
                    lateralFriction=0.1,
                    jointLowerLimit=(-np.pi/2 + anchor_angle),
                    jointUpperLimit=anchor_angle,
                    maxJointVelocity=max_anchor_vel)

    def start(self):
        N = self.N
        joint_id = self.joint_id
        ut = self.sim_utils
        maxForce = 1
        max_f_arr = (np.zeros(2) + maxForce).tolist()
        t = time.time()
        rsys = self.rsys
        while (1):
            is_done = rsys.step(rsys.x, rsys.prev_u, time.time())
            u = rsys.u
            x = rsys.x
            for i in range(N):
                r = self.rh[i]
                ak = r
                if self.is_anchor_separate:
                    ak = self.ah[i]

                # wheel control
                vl, vr = ut.get_lr(u[2*i], u[2*i+1])
                p.setJointMotorControlArray(r, 
                            [joint_id['left'], joint_id['right']],
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocities=[vl, vr],
                            forces=max_f_arr)

                # anchor control
                left_state = p.getJointState(ak, 
                                        joint_id['left_anchor'])[0]
                right_state = p.getJointState(ak, 
                                        joint_id['right_anchor'])[0]
                lv = ut.get_anchor_vel(left_state)
                rv = ut.get_anchor_vel(right_state, flip=-1)
                lf = ut.get_anchor_force(left_state)
                rf = ut.get_anchor_force(-right_state)
                p.setJointMotorControlArray(ak, 
                            [joint_id['left_anchor'], joint_id['right_anchor']],
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocities=[lv, rv],
                            forces=[lf, rf])

                pose, quat = p.getBasePositionAndOrientation(r)
                yaw = p.getEulerFromQuaternion(quat)[2]
                x[3*i:(3*i+2)] = [pose[0], pose[1]]
                x[3*i+2] = yaw

            rsys.x = x
            p.stepSimulation()
            t_diff = time.time() - t

    def end(self):
        p.disconnect()

