#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert, Tomas

import copy
import os
import sys
import time
import numpy as np
import math

sys.path.append('../drl_agent/turtlebot3_drl')
from turtlebot3_drl.common.settings import ENABLE_VISUAL, ENABLE_STACKING, OBSERVE_STEPS, MODEL_STORE_INTERVAL
from turtlebot3_drl.common.storagemanager import StorageManager
from turtlebot3_drl.common.logger import Logger
from turtlebot3_drl.common import utilities as util

from turtlebot3_msgs.srv import DrlStep, Goal
from std_srvs.srv import Empty

import rclpy
from rclpy.node import Node
from turtlebot3_drl.common.replaybuffer import ReplayBuffer

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
# Add new msgs - McuPose
from firmware_msgs.msg import McuPose

from ddpg import DDPG
from rclpy.qos import qos_profile_sensor_data

from turtlebot3_drl.drl_environment.reward import SUCCESS, COLLISION_WALL, UNKNOWN

NUM_SCAN_SAMPLES = 40
NUM_SCAN_SAMPLES_REAL = 720
LINEAR = 0
ANGULAR = 1
ENABLE_DYNAMIC_GOALS = False

ACTION_LINEAR_MAX   =  2.5 # in m/s
ACTION_ANGULAR_MAX  = -4.5

# in meters
ROBOT_MAX_LIDAR_VALUE   = 16
MAX_LIDAR_VALUE         = 4.5

MINIMUM_COLLISION_DISTANCE  = 0.13
MINIMUM_GOAL_DISTANCE       = 0.14
OBSTACLE_RADIUS             = 0.16
MAX_NUMBER_OBSTACLES        = 0

X_GOAL = 2.4
Y_GOAL = -0.8

ARENA_LENGTH    = 3.2
ARENA_WIDTH     = 3.2
MAX_GOAL_DISTANCE = math.sqrt(ARENA_LENGTH**2 + ARENA_WIDTH**2)

ENABLE_BACKWARD = True


class DrlAgent(Node):
    def __init__(self, algorithm, training, load_session="", load_episode=0, train_stage=util.test_stage):
        super().__init__(algorithm + '_agent')
        self.algorithm = algorithm
        self.is_training = int(training)
        self.load_session = load_session
        self.episode = int(load_episode)
        self.train_stage = train_stage
        if (not self.is_training and not self.load_session):
            quit("ERROR no test agent specified")
        self.device = util.check_gpu()
        self.sim_speed = 0.1
        print(f"{'training' if (self.is_training) else 'testing' } on stage: {util.test_stage}")

        self.total_steps = 0
        self.observe_steps = OBSERVE_STEPS

        if self.algorithm == 'dqn':
            self.model = DQN(self.device, self.sim_speed)
        elif self.algorithm == 'ddpg':
            self.model = DDPG(self.device, self.sim_speed)
        elif self.algorithm == 'td3':
            self.model = TD3(self.device, self.sim_speed)
        else:
            quit(f"invalid algorithm specified: {self.algorithm}, chose one of: ddpg, td3, td3conv")

        self.replay_buffer = ReplayBuffer(self.model.buffer_size)


        # ===================================================================== #
        #                             Model loading                             #
        # ===================================================================== #

        self.sm = StorageManager(self.algorithm, self.train_stage, self.load_session, self.episode, self.device)

        if self.load_session:
            del self.model
            self.model = self.sm.load_model()
            self.model.device = self.device
            self.sm.load_weights(self.model.networks)
            self.sm.new_session_dir()
            self.sm.store_model(self.model)
            
        self.logger = Logger(self.is_training, self.sm.machine_dir, self.sm.session_dir, self.sm.session, self.model.get_model_parameters(), self.model.get_model_configuration(), str(util.test_stage), self.algorithm, self.episode)
        
        # ===================================================================== #
        #                              Topics and Env                           #
        # ===================================================================== #
        self.scan_topic = 'scan'
        self.vel_topic = 'mcu_vel'
        self.odom_topic = 'mcu_pose'

        self.goal_x, self.goal_y = X_GOAL, Y_GOAL
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_x_prev, self.robot_y_prev = 0.0, 0.0
        self.robot_heading = 0.0
        self.total_distance = 0.0
        self.robot_tilt = 0.0

        self.done = False
        self.succeed = UNKNOWN
        self.obstacle_distance = MAX_LIDAR_VALUE

        self.scan_ranges = [MAX_LIDAR_VALUE] * 720

        self.local_step = 0
        self.goal_distance = math.sqrt((X_GOAL - self.robot_x)**2 + (Y_GOAL - self.robot_y)**2)
        self.goal_angle    = 0.0

        self.cmd_vel_pub = self.create_publisher(Twist, self.vel_topic, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(McuPose, self.odom_topic, self.odom_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_profile_sensor_data)
        
        # ===================================================================== #
        #                             Start Process                             #
        # ===================================================================== #


    def get_normalized_scan(self):
        scan = []
        temp = int(NUM_SCAN_SAMPLES_REAL / 40)

        self.obstacle_distance = 1
        for i in range(NUM_SCAN_SAMPLES):
            min_temp = min(self.scan_ranges[i*temp: (i*temp+2)])
            scan_ind = np.clip(float(min_temp) / MAX_LIDAR_VALUE, 0, 1)
            scan.append( scan_ind)
            if scan_ind < self.obstacle_distance:
                self.obstacle_distance = scan_ind
        self.obstacle_distance *= MAX_LIDAR_VALUE

        return scan

    
    def scan_callback(self, msg):
        if len(msg.ranges) != NUM_SCAN_SAMPLES_REAL:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {NUM_SCAN_SAMPLES}")
        # noramlize laser values
        self.scan_ranges = msg.ranges
        print("LIDAR")

    
    def odom_callback(self, msg):
        self.robot_x = msg.x
        self.robot_y = -msg.y
        self.robot_heading = -msg.theta

        diff_y = self.goal_y - self.robot_y
        diff_x = self.goal_x - self.robot_x
        distance_to_goal = math.sqrt(diff_x**2 + diff_y**2)
        heading_to_goal = math.atan2(diff_y, diff_x)
        goal_angle = heading_to_goal - self.robot_heading

        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = distance_to_goal
        self.goal_angle = goal_angle
        print("ODOM")

    def get_state(self, action_linear_previous, action_angular_previous):
        state = copy.deepcopy(self.get_normalized_scan())# range: [ 0, 1]

        state.append(float(np.clip((self.goal_distance / MAX_GOAL_DISTANCE), 0, 1)))        # range: [ 0, 1]
        state.append(float(self.goal_angle) / math.pi)                                      # range: [-1, 1]
        state.append(float(action_linear_previous))                                         # range: [-1, 1]
        state.append(float(action_angular_previous))                                        # range: [-1, 1]
        self.local_step += 1

        if self.goal_distance < 0.15:
            print("Outcome: Goal reached! :)")
            self.succeed = SUCCESS
            self.cmd_vel_pub.publish(Twist())
            self.done = True
            return state

        if self.local_step > 15: # Grace period
            if self.obstacle_distance < MINIMUM_COLLISION_DISTANCE:
                print("Outcome: Collision! (wall) :(")
                self.succeed = COLLISION_WALL
                self.done = True
        return state
    
    def step_comm(self, action_current, action_past):	
        if len(action_current) == 0:
            state = self.get_state(0, 0)
            done = False
            success = UNKNOWN
            return state, done, success

        # Un-normalize actions
        if ENABLE_BACKWARD:
            action_linear = action_current[LINEAR] * ACTION_LINEAR_MAX
        else:
            action_linear = (action_current[LINEAR] + 1) / 2 * ACTION_LINEAR_MAX
        action_angular = action_current[ANGULAR]*ACTION_ANGULAR_MAX
        
        # Publish action cmd
        twist = Twist()
        twist.linear.x = action_linear
        twist.angular.z = action_angular
        self.cmd_vel_pub.publish(twist)
        
        # Prepare repsonse
        state = self.get_state(action_past[LINEAR], action_past[ANGULAR])
        done = self.done
        success = self.succeed
        if self.done:
            # Reset variables
            self.local_step = 0
        
        return state, done, success

    def process(self):
        # util.pause_simulation(self)
        while (True):
            episode_done = False
            step = 0
            action_past = [0.0, 0.0]
            state, _, _ = self.step_comm( [0.0, 0.0], [0.0, 0.0])

            time.sleep(0.5)

            while not episode_done:
                action = self.model.get_action(state, self.is_training, step, ENABLE_VISUAL)

                action_current = action

                next_state, episode_done, sucess = self.step_comm(action_current, action_past)
                if episode_done:
                    self.step_comm( [0.0, 0.0],[0.0, 0.0])
                    rclpy.shutdown()

                state = copy.deepcopy(next_state)
                print(action,episode_done)
                
                time.sleep(0.005)

            self.total_steps += step

def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    drl_agent = DrlAgent(*args)
    rclpy.spin(drl_agent)
    drl_agent.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
