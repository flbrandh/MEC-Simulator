# Copyright (C) 2018 Florian Brandherm
# This file is part of flbrandh/MEC-Simulator <https://github.com/flbrandh/MEC-Simulator>.
#
# flbrandh/MEC-Simulator is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# flbrandh/MEC-Simulator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with flbrandh/MEC-Simulator.  If not, see <http://www.gnu.org/licenses/>.

import simulator, abc
import migrationAlgorithmInterface
import tensorflow.keras as K
import numpy as np
import pprint
import random
from cloudNeighborhood import *


class DQNMigrationAlgorithm(migrationAlgorithmInterface.MigrationAlgorithm):

    class Hyperparameters:
        def __init__(self):
            # problem-posing-related:
            self.max_num_neighbor_clouds = 10
            self.max_num_services = 10
            self.do_training = True
            # NN-related:
            self.model = None #None: new model, acoding to the below parameters. String: JSON of a model.
            self.network_width = 50
            self.network_depth = 7
            self.discount_factor = 0.9
            self.max_replay_memory_size = 10000
            # training-related
            self.episode_length = 100
            self.epsilon = 0.1
            self.initial_exploration_boost = 1000
            self.batch_fraction_of_replay_memory = 0.1
            self.num_epochs = 50
            self.target_model_update_frequency = 1 #1: no target network

        def get_serializable_dict(self):
            return self.__dict__

    def __init__(self, hyperparameters, rng = random.Random(), verbose= False):
        self.hyperparameters = hyperparameters
        self.rng = rng
        self.shared_agent = DQNMigrationAlgorithm.Agent(hyperparameters, rng, verbose)

    def create_instance(self, cloud): #TODO think about hew to rudimentarily model the communication between instances and the learner (only in the final system)
        return DQNMigrationAlgorithm.Instance(cloud, self.shared_agent, self.hyperparameters)

    def get_name(self):
        return 'DQNMigrationAlgorithm'

    class Agent:

        class QSample:
            """
            Represents one sample of the q-function : (s,a,r,s',[a'])
            """
            def __init__(self, state_features, action_features, reward, next_state_features, possible_next_action_features):
                self.state_features = state_features
                self.action_features = action_features
                self.reward = reward
                self.next_state_features = next_state_features
                self.possible_next_action_features = possible_next_action_features

        def __init__(self, hyperparameters, rng, verbose):
            self.rng = rng
            self.verbose = verbose
            self.hyperparameters = hyperparameters

            self.Q_model = self.build_model()
            self.Q_target_model = self.build_model()
            
            self.episode=0
            self.iteration=0

            self.last_service = None
            self.last_state_features = {}
            self.last_action_features = {}
            self.last_reward = {}

            self.replayMemory = []
            #statistics:
            self.totalEpisodeReward=0
            self.avgRewards = []
            self.maxEpisodeReward = -math.inf
            self.maxRewards = []
            self.minEpisodeReward = math.inf
            self.minRewards = []


        def build_model(self):
            if self.hyperparameters.model:
                model =  K.models.model_from_json(self.hyperparameters.model)
            else:
                state_action_features = K.Input(shape=(len(self.state_features())+len(self.action_features()),))
                x = state_action_features
                for i in range(7):
                    x = K.layers.Dense(50, activation='relu')(x)

                Q = K.layers.Dense(1)(x)
                model = K.Model(
                    inputs=state_action_features,
                    outputs=Q)
            # compile model
            model.compile(loss='mean_squared_error',
                                 optimizer='adam')
            return model

        def serialize_model(self):
            return self.Q_model.get_weights()

        def deserialize_model(self, weights):
            self.Q_model.set_weights(weights)

        def state_features(self, src_cloud = None, neighbor_clouds = None):
            features =  self.cloud_features(src_cloud)
            features += self.cloud_service_features(src_cloud)
            features += self.neighbor_features(neighbor_clouds)
            return features

        def neighbor_features(self, neighbor_clouds = None):
            if neighbor_clouds:
                features = []
                for i in range(self.hyperparameters.max_num_neighbor_clouds):
                    if i < len(neighbor_clouds):
                        features+=self.cloud_features(neighbor_clouds[i])
                    else:
                        features+=self.cloud_features()
                return features
            else:
                return self.hyperparameters.max_num_neighbor_clouds * self.cloud_features()

        def action_features(self, stop=False, dst_cloud = None, service = None):
            features =  [int(stop)]+self.cloud_features(dst_cloud)+self.service_features(service)
            return features

        def cloud_service_features(self, cloud = None):
            if cloud:
                services = cloud.get_services()
                features = []
                for i in range(self.hyperparameters.max_num_services):
                    if i < len(services):
                        features+=self.service_features(services[i])
                    else:
                        features+=self.service_features()
                return features
            else:
                return self.hyperparameters.max_num_services * self.service_features()


        def cloud_features(self, cloud=None):
            if cloud:
                return [cloud.memory_capacity()/10,
                        cloud.totalMemoryRequirement()/10,
                        cloud.get_node().get_pos()[0],
                        cloud.get_node().get_pos()[1]]
            else:
                return [0] * 4

        def service_features(self, service=None):
            if service:
                features = [service.is_active(),
                            service.get_memory_requirement()/10,
                            service.get_latency_requirement()/10,
                            service.get_user().get_base_station().get_pos()[0],
                            service.get_user().get_base_station().get_pos()[1]]
                last_cloud = service.get_last_cloud()
                if last_cloud:
                    features += [last_cloud.get_pos()[0], last_cloud.get_pos()[1]]
                else:
                    features += [0, 0]
                return features
            else:
                return [0] * 7

        def process_migration_event(self, service, cloud, cloud_neighborhood):
            """
            determines migration actions for a cloud at a migration event
            :param cloud: the cloud that this function was invoked for
            :return: Migration Result
            """

            neighbor_clouds = cloud_neighborhood.get_neighboring_clouds()
            assert len(neighbor_clouds) <= self.hyperparameters.max_num_neighbor_clouds

            state_features = self.state_features(cloud)

            predicted_action_values = np.zeros((len(neighbor_clouds),))
            possible_action_features = []
            for ci, neighbor_cloud in enumerate(neighbor_clouds):
                action_features = self.action_features(False, neighbor_cloud, service)
                state_action_features = state_features + action_features
                possible_action_features.append(action_features)
                pred = self.Q_model.predict(np.array([state_action_features]))
                predicted_action_values[ci] = pred
            predicted_stop_action_value = self.Q_model.predict(np.array([self.state_features() + self.action_features(True)]))[0]
            possible_action_features.append(self.action_features(True,cloud,service))

            #epsilon-greedy policy:
            epsilon = self.hyperparameters.epsilon
            if self.hyperparameters.initial_exploration_boost != 0:
                epsilon += (1-self.hyperparameters.epsilon)*math.exp(-self.iteration/self.hyperparameters.initial_exploration_boost)

            stop_action = False
            if self.rng.random() < epsilon:
                #random choice:
                if self.rng.random() < 1/(len(neighbor_clouds)+1):
                    stop_action = True
                else:
                    ci = self.rng.randint(0,len(neighbor_clouds)-1)
            else:
                # choose action with maximum state-action-value
                max_state_action_value = -math.inf

                ci = 0
                for cii in range(len(neighbor_clouds)):
                    if predicted_action_values[cii] > max_state_action_value:
                        max_state_action_value = predicted_action_values[cii]
                        ci = cii
                if predicted_stop_action_value > max_state_action_value:
                    max_state_action_value = predicted_stop_action_value
                    stop_action = True

            # initialize cache if necessary
            if service not in self.last_state_features:
                self.last_state_features[service] = None
                self.last_action_features[service] = None
                self.last_reward[service] = None

            # enter sample into replay memory:
            self.last_service = service
            if self.last_state_features[service]:
                assert self.last_action_features[service] != None
                assert self.last_reward[service] != None
                qsample = DQNMigrationAlgorithm.Agent.QSample(
                    self.last_state_features[service], self.last_action_features[service],
                    self.last_reward[service], state_features, possible_action_features)
                self.replayMemory.append(qsample)
                if len(self.replayMemory) > self.hyperparameters.max_replay_memory_size:
                    self.replayMemory.pop(0)
            self.last_state_features[service] = state_features
            self.last_action_features[service] = self.action_features(True,cloud,service)
            if not stop_action:
                self.last_action_features[service] = self.action_features(False, neighbor_clouds[ci], service)


            #train the network if the episode length has passed
            if self.hyperparameters.do_training and 0 == self.iteration % self.hyperparameters.episode_length:
                minibatch_size = int(self.hyperparameters.batch_fraction_of_replay_memory*len(self.replayMemory))
                if self.verbose:
                    print('minibatch_size: ',minibatch_size,' replay memory size:', len(self.replayMemory))
                if minibatch_size > 10: #it's not worth it below that
                    minibatch = self.rng.sample(range(len(self.replayMemory)),minibatch_size)

                    # construct x
                    x = np.zeros((minibatch_size,
                                  len(self.state_features()) + len(
                                      self.action_features())))
                    for i, batch_sample in enumerate(minibatch):
                        qsample = self.replayMemory[batch_sample]
                        x[i, :] = np.array(
                            qsample.state_features + qsample.action_features)

                    # construct y (discounted rewards), using the taget model
                    y = np.zeros(minibatch_size)
                    for i, batch_sample in enumerate(minibatch):
                        qsample = self.replayMemory[batch_sample]
                        state_features = qsample.state_features
                        max_state_action_value = -math.inf
                        for action_features in qsample.possible_next_action_features:
                            prediction = self.Q_target_model.predict(np.array([state_features + action_features]))[0]
                            max_state_action_value = max(max_state_action_value, prediction)
                        y[i] = qsample.reward + self.hyperparameters.discount_factor * max_state_action_value

                    # train
                    self.Q_model.fit(x,y,epochs=self.hyperparameters.num_epochs, batch_size=32,verbose=0)

                    if self.verbose:
                        print('iteration=',self.iteration,', epsilon=',epsilon, ', discount factor=',self.hyperparameters.discount_factor)

                    if 0 == self.episode % self.hyperparameters.target_model_update_frequency:
                        self.Q_target_model.set_weights(self.Q_model.get_weights())
                        if self.verbose:
                            print('**** Updated the target model in iteration '+str(self.iteration)+'. ****')

                    self.episode += 1

            #construct returned migration actions
            if stop_action:
                #do nothing if an action is invalid of the null action
                return migrationAlgorithmInterface.NoMigrationAction(service, cloud)
            else:
                target_cloud = neighbor_clouds[ci]
                return migrationAlgorithmInterface.MigrationAction(service, cloud, target_cloud)


        def give_reward(self, reward):
            """
            This function gives an immediate reward for the last action. If this method isn't called, the reward for the last action is 0.
            :param reward:
            :return: None
            """
            service = self.last_service
            self.last_reward[service] = reward
            self.totalEpisodeReward+=reward
            self.maxEpisodeReward = max(self.maxEpisodeReward, reward)
            self.minEpisodeReward = min(self.minEpisodeReward, reward)

            if self.hyperparameters.episode_length-1 == self.iteration % self.hyperparameters.episode_length:

                # track statistics
                self.avgRewards.append(self.totalEpisodeReward/self.hyperparameters.episode_length)
                self.maxRewards.append(self.maxEpisodeReward)
                self.minRewards.append(self.minEpisodeReward)
                self.totalEpisodeReward=0
                self.maxEpisodeReward = -math.inf
                self.minEpisodeReward = math.inf


            self.iteration+=1

    class Instance(migrationAlgorithmInterface.MigrationAlgorithm.Instance):
        def __init__(self, cloud, shared_agent, hyperparameters):
            self.__cloud = cloud
            self.__shared_agent = shared_agent
            self.__cloud_neighborhood = KnnCloudNeighborhood(cloud, hyperparameters.max_num_neighbor_clouds)

        def process_migration_event(self, service):
            return self.__shared_agent.process_migration_event(service,self.__cloud,self.__cloud_neighborhood)

        def get_neighbor_clouds(self):
            return self.__cloud_neighborhood.get_neighboring_clouds()

        def give_reward(self, reward):
            self.__shared_agent.give_reward(reward)