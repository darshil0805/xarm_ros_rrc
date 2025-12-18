#!/usr/bin/env python3

import rospy
import cv2
import json
import base64
import numpy as np
import websocket
import threading
import time
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from dynamixel_sdk import * # Uses Dynamixel SDK library

class DynamixelGripper:
    def __init__(self, port='/dev/ttyUSB0', baudrate=57600, dxl_id=1):
        self.port = port
        self.baudrate = baudrate
        self.dxl_id = dxl_id
        
        # Control table addresses (from C++ implementation)
        self.ADDR_OPERATING_MODE = 11
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_GOAL_VELOCITY = 104
        self.ADDR_PRESENT_POSITION = 132
        self.PROTOCOL_VERSION = 2.0
        
        self.port_handler = PortHandler(self.port)
        self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)
        
        if not self.port_handler.openPort():
            rospy.logerr("Failed to open Dynamixel port")
            return
            
        if not self.port_handler.setBaudRate(self.baudrate):
            rospy.logerr("Failed to set Dynamixel baudrate")
            return

        # Set Velocity Control Mode (1)
        self.packet_handler.write1ByteTxRx(self.port_handler, self.dxl_id, self.ADDR_OPERATING_MODE, 1)
        # Enable Torque
        self.packet_handler.write1ByteTxRx(self.port_handler, self.dxl_id, self.ADDR_TORQUE_ENABLE, 1)
        
        rospy.loginfo("Dynamixel Gripper initialized")

    def get_position(self):
        dxl_present_position, dxl_comm_result, dxl_error = self.packet_handler.read4ByteTxRx(
            self.port_handler, self.dxl_id, self.ADDR_PRESENT_POSITION)
        return dxl_present_position

    def set_velocity(self, velocity):
        # Clamp velocity if needed, or handle limits based on position
        self.packet_handler.write4ByteTxRx(self.port_handler, self.dxl_id, self.ADDR_GOAL_VELOCITY, int(velocity))

    def close(self):
        self.packet_handler.write1ByteTxRx(self.port_handler, self.dxl_id, self.ADDR_TORQUE_ENABLE, 0)
        self.port_handler.closePort()

class PolicyInferenceNode:
    def __init__(self):
        rospy.init_node('policy_inference_node')
        
        # Parameters
        self.server_url = rospy.get_param('~server_url', 'ws://localhost:8765')
        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.joint_state_topic = rospy.get_param('~joint_state_topic', '/joint_states')
        
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_joints = None
        self.lock = threading.Lock()
        
        # ROS Subscribers
        rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback)
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.joint_state_topic, JointState, self.joint_callback)
        
        # ROS Publishers for action execution (using Servo inputs)
        self.joint_pub = rospy.Publisher('/servo_server/delta_joint_cmds', JointJog, queue_size=1)
        
        # Dynamixel Gripper
        self.gripper = DynamixelGripper()
        
        # WebSocket Client
        self.ws = None
        self.ws_thread = threading.Thread(target=self.init_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        rospy.loginfo("Policy Inference Node started")

    def rgb_callback(self, data):
        with self.lock:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")

    def depth_callback(self, data):
        with self.lock:
            self.latest_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")

    def joint_callback(self, data):
        with self.lock:
            # Assume joint order matches policy expectations
            self.latest_joints = list(data.position)

    def init_websocket(self):
        while not rospy.is_shutdown():
            try:
                self.ws = websocket.WebSocketApp(self.server_url,
                                                 on_message=self.on_message,
                                                 on_error=self.on_error,
                                                 on_close=self.on_close)
                self.ws.on_open = self.on_open
                self.ws.run_forever()
            except Exception as e:
                rospy.logerr(f"WebSocket error: {e}")
            time.sleep(5)

    def on_open(self, ws):
        rospy.loginfo("Connected to policy server")
        # Start a loop to send states
        threading.Thread(target=self.state_sender_loop).start()

    def on_message(self, ws, message):
        data = json.loads(message)
        if 'actions' in data:
            self.execute_action_chunk(data['actions'])

    def on_error(self, ws, error):
        rospy.logerr(f"WS Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        rospy.loginfo("WS Closed")

    def state_sender_loop(self):
        rate = rospy.Rate(10) # 10Hz or as needed
        while not rospy.is_shutdown() and self.ws.sock and self.ws.sock.connected:
            with self.lock:
                if self.latest_rgb is not None and self.latest_joints is not None:
                    # Encode images
                    _, buffer = cv2.imencode('.jpg', self.latest_rgb)
                    rgb_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    depth_payload = None
                    if self.latest_depth is not None:
                        # Normalize depth for encoding or send as is
                        depth_payload = self.latest_depth.tolist() # Example, might be too large
                    
                    payload = {
                        'rgb': rgb_base64,
                        'joints': self.latest_joints,
                        'gripper': self.gripper.get_position()
                    }
                    self.ws.send(json.dumps(payload))
            rate.sleep()

    def execute_action_chunk(self, actions):
        rospy.loginfo(f"Executing chunk of {len(actions)} actions")
        for action in actions:
            if rospy.is_shutdown():
                break
                
            # action format: [j1, j2, j3, j4, j5, j6, gripper]
            # Here we use delta commands for Servo
            joint_cmd = JointJog()
            joint_cmd.header.stamp = rospy.Time.now()
            joint_cmd.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
            
            # This is a simplification: assuming action provides absolute targets or deltas
            # If the policy outputs absolute, we should calculate delta or use a different interface
            # For now, let's assume deltas for demonstration:
            joint_cmd.velocities = action[:6]
            self.joint_pub.publish(joint_cmd)
            
            # Gripper control (velocity)
            self.gripper.set_velocity(action[6])
            
            # Control frequency for the chunk
            time.sleep(0.1) 

    def run(self):
        rospy.spin()
        if self.ws:
            self.ws.close()
        self.gripper.close()

if __name__ == '__main__':
    try:
        node = PolicyInferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
