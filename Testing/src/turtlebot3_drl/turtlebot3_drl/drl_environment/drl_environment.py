#
#!/usr/bin/env python3
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

import math
import numpy
import sys
import copy
from numpy.core.numeric import Infinity

from geometry_msgs.msg import Pose, Twist
from rosgraph_msgs.msg import Clock
#from nav_msgs.msg imaport Odometry
from sensor_msgs.msg import LaserScan
# Add new msgs - McuPose
from firmware_msgs.msg import McuPose

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from turtlebot3_msgs.srv import DrlStep, Goal, RingGoal

from turtlebot3_drl.common.settings import ENABLE_BACKWARD, EPISODE_TIMEOUT_SECONDS, ENABLE_MOTOR_NOISE

from turtlebot3_drl.drl_environment.reward import UNKNOWN, SUCCESS, COLLISION_WALL, COLLISION_OBSTACLE, TIMEOUT, TUMBLE

NUM_SCAN_SAMPLES = 40
NUM_SCAN_SAMPLES_REAL = 720
LINEAR = 0
ANGULAR = 1
ENABLE_DYNAMIC_GOALS = False

ACTION_LINEAR_MAX   =  12.05  # in m/s
ACTION_ANGULAR_MAX  =  70

# in meters
ROBOT_MAX_LIDAR_VALUE   = 18
MAX_LIDAR_VALUE = 3.85

MINIMUM_COLLISION_DISTANCE  = 0.13
MINIMUM_GOAL_DISTANCE       = 0.15
OBSTACLE_RADIUS             = 0.16
MAX_NUMBER_OBSTACLES        = 0

X_GOAL = 2.4
Y_GOAL = 0.0

ARENA_LENGTH    = 4.2
ARENA_WIDTH     = 4.2
MAX_GOAL_DISTANCE = math.sqrt(ARENA_LENGTH**2 + ARENA_WIDTH**2)

ENABLE_BACKWARD = True

X_GOALS = [-1.6, -1.6, -0.8, 0.0]
Y_GOALS = [1.6, 0.0, -2.4, 0.0]

WAIT_NEW_GOAL = 7

results = []

# EPISODE_BASE_DEADLINE_SECONDS = 15 if ENABLE_DYNAMIC_GOALS else 50
class DRLEnvironment(Node):
    def __init__(self):
        super().__init__('drl_environment')
        self.stage = 9

        print(f"running on stage: {self.stage}")
        self.episode_timeout = EPISODE_TIMEOUT_SECONDS

        self.scan_topic = 'scan'
        self.vel_topic = 'mcu_vel'
        self.goal_topic = 'goal_pose'
        self.odom_topic = 'mcu_pose'

        self.goal_index = 0
        self.new_goal_clocks = 0
        self.goal_x, self.goal_y = X_GOALS[0], Y_GOALS[0]
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_x_prev, self.robot_y_prev = 0.0, 0.0
        self.robot_heading = 0.0
        self.total_distance = 0.0
        self.robot_tilt = 0.0

        self.done = False
        self.succeed = UNKNOWN
        self.episode_deadline = Infinity
        self.reset_deadline = False
        self.clock_msgs_skipped = 0

        self.obstacle_distances = [Infinity] * MAX_NUMBER_OBSTACLES

        self.new_goal = False
        self.goal_angle = 0.0
        self.goal_distance = MAX_GOAL_DISTANCE
        self.initial_distance_to_goal = MAX_GOAL_DISTANCE

        self.scan_ranges = [MAX_LIDAR_VALUE] * NUM_SCAN_SAMPLES
        self.obstacle_distance = MAX_LIDAR_VALUE

        self.difficulty_radius = 1
        self.local_step = 0
        self.time_sec = 0
        
        self.diff_x = 0
        self.diff_y = 0

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=4)
        qos_odom = qos_profile_sensor_data
        qos_odom.depth = 6
        # publishers
        self.cmd_vel_pub = self.create_publisher(Twist, self.vel_topic, qos_profile_sensor_data)
        # subscribers
        #self.goal_pose_sub = self.create_subscription(Pose, self.goal_topic, self.goal_pose_callback, 10)
        self.odom_sub = self.create_subscription(McuPose, self.odom_topic, self.odom_callback, qos_profile=qos_profile_sensor_data) 
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_profile_sensor_data)
        #self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, qos_profile=qos_clock)
        #self.obstacle_odom_sub = self.create_subscription(Odometry, 'obstacle/odom', self.obstacle_odom_callback, qos)
        self.step_comm_server = self.create_service(DrlStep, 'step_comm', self.step_comm_callback)

        '''
        # clients
        self.task_succeed_client = self.create_client(RingGoal, 'task_succeed')
        self.task_fail_client = self.create_client(RingGoal, 'task_fail')
        # servers
        self.step_comm_server = self.create_service(DrlStep, 'step_comm', self.step_comm_callback)
        self.goal_comm_server = self.create_service(Goal, 'goal_comm', self.goal_comm_callback)
        '''
    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    '''
    def goal_pose_callback(self, msg):
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.new_goal = True

    def goal_comm_callback(self, request, response):
        response.new_goal = self.new_goal
        return response
    '''
        
    '''
    def obstacle_odom_callback(self, msg):
        if 'obstacle' in msg.child_frame_id:
            robot_pos = msg.pose.pose.position
            obstacle_id = int(msg.child_frame_id[-1]) - 1
            diff_x = self.robot_x - robot_pos.x
            diff_y = self.robot_y - robot_pos.y
            self.obstacle_distances[obstacle_id] = math.sqrt(diff_y**2 + diff_x**2)
        else:
            print("ERROR: received odom was not from obstacle!")
    '''
    def odom_callback(self, msg):
        self.robot_x = msg.x
        self.robot_y = -msg.y
        #_, _, self.robot_heading = util.euler_from_quaternion(msg.pose.pose.orientation)
        #self.robot_tilt = msg.pose.pose.orientation.y
        self.robot_heading = -msg.theta
        # calculate traveled distance for logging
        print(self.robot_x, self.robot_y)
        self.diff_y = self.goal_y - self.robot_y
        self.diff_x = self.goal_x - self.robot_x
        distance_to_goal = math.sqrt(self.diff_x**2 + self.diff_y**2)
        heading_to_goal = math.atan2(self.diff_y, self.diff_x)
        goal_angle = heading_to_goal - self.robot_heading

        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = distance_to_goal
        self.goal_angle = goal_angle
     
    def scan_callback(self, msg):
        if len(msg.ranges) != NUM_SCAN_SAMPLES_REAL:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {NUM_SCAN_SAMPLES}")
            return
        # noramlize laser values
        temp = int(NUM_SCAN_SAMPLES_REAL / 40)
        self.obstacle_distance = 1
        for i in range(NUM_SCAN_SAMPLES):
		#temp = (i+1) * (NUM_SCAN_SAMPLES_REAL / 40)
                min_temp = min(msg.ranges[i*temp: (i*temp+2)])
                self.scan_ranges[i] = numpy.clip(float(min_temp) / MAX_LIDAR_VALUE, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    	self.obstacle_distance = self.scan_ranges[i]
        self.obstacle_distance *= MAX_LIDAR_VALUE
    '''
    def clock_callback(self, msg):
        self.time_sec = msg.clock.sec
        if self.reset_deadline:
            self.clock_msgs_skipped += 1
            if self.clock_msgs_skipped > 10: # Wait a few message for simulation to reset clock
                episode_time = self.episode_timeout
                if ENABLE_DYNAMIC_GOALS:
                    episode_time = numpy.clip(episode_time * self.difficulty_radius, 10, 50)
                self.episode_deadline = self.time_sec + episode_time
                self.reset_deadline = False
                self.clock_msgs_skipped = 0
    '''
    '''
    def stop_reset_robot(self, success):
        self.cmd_vel_pub.publish(Twist()) # stop robot
        self.episode_deadline = Infinity
        self.done = True
        req = RingGoal.Request()
        req.robot_pose_x = self.robot_x
        req.robot_pose_y = self.robot_y
        req.radius = numpy.clip(self.difficulty_radius, 0.5, 4)
        if success:
            self.difficulty_radius *= 1.01
            while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('success service not available, waiting again...')
            self.task_succeed_client.call_async(req)
        else:
            self.difficulty_radius *= 0.99
            while not self.task_fail_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('fail service not available, waiting again...')
            self.task_fail_client.call_async(req)
    '''
    def rotate(self, arr, n):
        return arr[n:] + arr[:n]
    def get_state(self, action_linear_previous, action_angular_previous):
        state = copy.deepcopy(self.scan_ranges)# range: [ 0, 1]

        state.append(float(numpy.clip((self.goal_distance / MAX_GOAL_DISTANCE), 0, 1)))     # range: [ 0, 1]
        state.append(float(self.goal_angle) / math.pi)                                      # range: [-1, 1]
        state.append(float(action_linear_previous))                                         # range: [-1, 1]
        state.append(float(action_angular_previous))                                        # range: [-1, 1]
        self.local_step += 1
       # print(self.goal_distance)
        #print(self.robot_x, self.robot_y) 
        if self.goal_distance < 0.1:
            
            self.cmd_vel_pub.publish(Twist())
            
            if (self.goal_index == 3):
                print("Outcome: Goal reached! :)")
                self.succeed = SUCCESS
                if self.done:
                    results.append('{}, {}'.format(self.robot_x, self.robot_y))
                    print(results)
                    rclpy.shutdown()
                self.done = True
                return state

            if (self.new_goal_clocks < 300):
                self.new_goal_clocks += 1
                self.succeed = WAIT_NEW_GOAL
                self.done = False
                return state  
            
            results.append('{}, {}'.format(self.robot_x, self.robot_y))
            self.succeed = UNKNOWN
            self.goal_index += 1
            self.new_goal_clocks = 0
            self.goal_x = X_GOALS[self.goal_index]
            self.goal_y = Y_GOALS[self.goal_index]
            self.goal_distance = MAX_GOAL_DISTANCE

            return state

        if self.local_step > 15: # Grace period
            # Success
            if self.obstacle_distance < MINIMUM_COLLISION_DISTANCE:
                print("Outcome: Collision! (wall) :(")
                self.succeed = COLLISION_WALL
                self.done = True
        return state

    def initalize_episode(self, response):
        self.initial_distance_to_goal = self.goal_distance
        response.state = self.get_state(0, 0)
        response.reward = 0.0
        response.done = False
        response.distance_traveled = 0.0
        return response

    def step_comm_callback(self, request, response):	
        if len(request.action) == 0:
            return self.initalize_episode(response)

        # Un-normalize actions
        if ENABLE_BACKWARD:
            action_linear = request.action[LINEAR] * ACTION_LINEAR_MAX
        else:
            action_linear = (request.action[LINEAR] + 1) / 2 * ACTION_LINEAR_MAX
        action_angular = request.action[ANGULAR]*ACTION_ANGULAR_MAX
       
        #print(action_linear, action_angular)
        # Publish action cmd
        if (self.succeed != WAIT_NEW_GOAL):
            twist = Twist()
            twist.linear.x = action_linear
            twist.angular.z = action_angular
            self.cmd_vel_pub.publish(twist)
        else:
            self.cmd_vel_pub.publish(Twist())

        # Prepare repsonse
        response.state = self.get_state(request.previous_action[LINEAR], request.previous_action[ANGULAR])
        response.reward = 0.0
        response.done = self.done
        response.success = self.succeed
        response.distance_traveled = 0.0
        if self.done:
            response.distance_traveled = self.total_distance
            # Reset variables
            self.total_distance = 0.0
            self.local_step = 0
            self.reset_deadline = True
        
        return response

def main(args=None):
    rclpy.init(args=args)
    drl_environment = DRLEnvironment()
    rclpy.spin(drl_environment)
    drl_environment.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
