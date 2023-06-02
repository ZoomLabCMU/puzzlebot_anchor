import sys
import time
import socket
import rospy
import threading
from collections import deque
import numpy as np
import random

class ClientThread(threading.Thread):
    def __init__(self, clientAddress, clientsocket, data_q):
        threading.Thread.__init__(self)
        self.dq = deque(maxlen=1)
        self.data_q = data_q
        self.csocket = clientsocket
        self.csocket.settimeout(0.05)
        # self.csocket.setblocking(0)
        self.clientAddress = clientAddress

        self.dt = 0
        self.prev_time = time.time()
        self.end_sig = False
        print ("New connection added: ", clientAddress)

    def run(self):
        print ("Connection from : ", self.clientAddress)
        while True:
            if self.end_sig: return
            try:
                if not self.dq: 
                    continue
                val = self.dq.pop()
                self.csocket.send(val.encode())
                rospy.loginfo("sent msg to %s:%d, %s" % (self.clientAddress[0], self.clientAddress[1], val))
            except Exception as e:
                print(e)

class TCPBridge:
    def __init__(self, N):
        self.N = N
        self.TCP_IP = '0.0.0.0'
        self.TCP_PORT = 8080 
        self.BUFFER_SIZE = 9

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((self.TCP_IP, self.TCP_PORT))
        self.threads = []
        self.robot_ips = []
        self.dts = []
    
    def start_listen(self):
        for i in range(self.N):

            rospy.loginfo("waiting for %d th robot" % i)
            self.s.listen(2)
            clientsock, clientAddress = self.s.accept()

            # process robot ips
            try:
                ip = int(clientAddress[0].split('.')[-1])
                self.robot_ips.append(ip)
            except:
                raise ValueError("Incoming IP not recognized.")
            data_q = deque(maxlen=1)
            newthread = ClientThread(clientAddress, clientsock, data_q)
            newthread.start()
            self.threads.append(newthread)
            rospy.loginfo("%d robot connected" % len(self.threads))
            self.dts.append(0)

        time.sleep(2)

    def send(self, tw):
        # tw is 2-by-N
        for i in range(self.N):
            [lv, rv] = tw[:, i]
            msg = "%04d,%04d\n" % (lv, rv) # left vel, right vel

            if not self.threads[i].is_alive(): continue
            self.threads[i].dq.append(msg)
            self.dts[i] = self.threads[i].dt

    def end(self):
        self.s.close()

