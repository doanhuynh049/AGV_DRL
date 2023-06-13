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

sys.path.append('../drl_agent/turtlebot3_drl')
from turtlebot3_drl.common.settings import ENABLE_VISUAL, ENABLE_STACKING, OBSERVE_STEPS, MODEL_STORE_INTERVAL
from turtlebot3_drl.common.storagemanager import StorageManager
from turtlebot3_drl.common.logger import Logger
from turtlebot3_drl.common import utilities as util


from ddpg import DDPG


from turtlebot3_msgs.srv import DrlStep, Goal
from std_srvs.srv import Empty

import rclpy
from rclpy.node import Node
from turtlebot3_drl.common.replaybuffer import ReplayBuffer

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
        if ENABLE_VISUAL:
            self.visual = DrlVisual(self.model.state_size, self.model.hidden_size)
            self.model.attach_visual(self.visual)
        # ===================================================================== #
        #                             Start Process                             #
        # ===================================================================== #

        self.step_comm_client = self.create_client(DrlStep, 'step_comm')
        self.process()


    def process(self):
        # util.pause_simulation(self)
        while (True):
            episode_done = False
            step, reward_sum, loss_critic, loss_actor = 0, 0, 0, 0
            action_past = [0.0, 0.0]
            state = util.init_episode(self)

            time.sleep(0.5)

            while not episode_done:
                action = self.model.get_action(state, self.is_training, step, ENABLE_VISUAL)

                action_current = action

                next_state, reward, episode_done, outcome, distance_traveled = util.step(self, action_current, action_past)
                if episode_done:
                    util.step(self, [0.0, 0.0],[0.0, 0.0])
                    rclpy.shutdown()

                if outcome == 7:
                    action_current = [0.0, 0.0]

                state = copy.deepcopy(next_state)
                action_past = copy.deepcopy(action_current)

                print(action,episode_done)
                
                time.sleep(0.05)

            self.total_steps += step

    def finish_episode(self, step, eps_duration, outcome, dist_traveled, reward_sum, loss_critic, lost_actor):
            print(f"Epi: {self.episode} R: {reward_sum:.2f} outcome: {util.translate_outcome(outcome)} \
                    steps: {step} steps_total: {self.total_steps}, time: {eps_duration:.2f}")

def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    drl_agent = DrlAgent(*args)
    rclpy.spin(drl_agent)
    drl_agent.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
