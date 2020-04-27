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

from migrationAlgorithmInterface import *
import cloudNeighborhood
import math
import numpy as np

class RewardAggregatorAgent:
    def __init__(self):
        self.avgRewards = []
        self.rewardBuffer = []

    def addReward(self, reward):
        self.rewardBuffer.append(reward)
        if len(self.rewardBuffer) >= 100:
            self.avgRewards.append(np.mean(self.rewardBuffer))
            self.rewardBuffer = []

class NeverMigrateAlgorithm(MigrationAlgorithm):
    """
    Never migrates any services.
    """
    def __init__(self):
        self.shared_agent = RewardAggregatorAgent()

    def get_name(self):
        return "never_migrate"

    def create_instance(self, cloud):
        return NeverMigrateAlgorithm.Instance(cloud, self.shared_agent)

    class Instance(MigrationAlgorithm.Instance):
        def __init__(self, cloud, reward_aggregator):
            self.__cloud = cloud
            self.__reward_aggregator = reward_aggregator

        def process_migration_event(self, service):
            return NoMigrationAction(service, self.__cloud)

        def give_reward(self, reward):
            self.__reward_aggregator.addReward(reward)

class AlwaysMigrateAlgorithm(MigrationAlgorithm):
    """
    Always migrates services to the cloud that has the lowest latency to its user (Assumes full connectivity!)
    Note that this will only work with the simulation since it requires global knowledge about the entire topology.
    """
    def __init__(self, only_available_clouds):
        self.__all_clouds = None
        self.__sim = [None]
        self.shared_agent = RewardAggregatorAgent()
        self.__only_available_clouds = only_available_clouds

    def get_name(self):
        return "always_migrate_global"

    def set_simulation(self,sim):
        self.__sim[0] = sim

    def create_instance(self, cloud):
        if not self.__all_clouds:
            self.__all_clouds = cloudNeighborhood.AllConnectedCloudNeighborhood(cloud).get_neighboring_clouds()
            self.__all_clouds.append(cloud)
        return AlwaysMigrateAlgorithm.Instance(self.__sim, cloud, self.__all_clouds, self.shared_agent, self.__only_available_clouds)

    class Instance(MigrationAlgorithm.Instance):
        def __init__(self,sim, cloud, all_clouds, reward_aggregator, only_available_clouds):
            self.__cloud = cloud
            self.__all_clouds = all_clouds
            self.__reward_aggregator = reward_aggregator
            self.__sim = sim
            self.__only_available_clouds = only_available_clouds

        def process_migration_event(self, service):
            migration_actions = []
            if service.lastUserNode is not service.user.get_base_station():
                target_cloud = None
                if self.__only_available_clouds:
                    target_cloud = self.__sim[0].closestCloud(service.user.get_base_station(),lambda cloud: (cloud.totalMemoryRequirement()+service.get_memory_requirement()) / cloud.memory_capacity() <=1)
                else:
                    target_cloud = self.__sim[0].closestCloud(service.user.get_base_station())
                if target_cloud is not self.__cloud: # don't migrate to yourself
                    return MigrationAction(service, self.__cloud, target_cloud)
            return NoMigrationAction(service, self.__cloud)

        def give_reward(self, reward):
            self.__reward_aggregator.addReward(reward)

class AlwaysMigrateToClosestInNeighborhoodAlgorithm(MigrationAlgorithm):
    """
    Always migrates services to the cloud that has the lowest latency to its user (Assumes full connectivity!)
    Note that this will only work with the simulation since it requires global knowledge about the entire topology.
    """
    def __init__(self, num_neighbors, only_available_clouds, distance_measure = 'euclidian'):
        self.__num_neighbors = num_neighbors
        self.__distance_measure = distance_measure
        if distance_measure not in ['euclidian', 'hops']:
            raise ValueError("invalid distance measure")
        self.shared_agent = RewardAggregatorAgent()
        self.__only_available_clouds = only_available_clouds

    def get_name(self):
        return "always_migrate_local_"+ self.__distance_measure

    def create_instance(self, cloud):
        if self.__distance_measure == 'euclidian':
            return AlwaysMigrateToClosestInNeighborhoodAlgorithm.InstanceEuclidian(cloud, self.__num_neighbors, self.shared_agent, self.__only_available_clouds)
        elif self.__distance_measure == 'hops':
            return AlwaysMigrateToClosestInNeighborhoodAlgorithm.InstanceHops(cloud, self.__num_neighbors, self.shared_agent, self.__only_available_clouds)

    class InstanceEuclidian(MigrationAlgorithm.Instance):
        def __init__(self, cloud, num_neighbors, reward_aggregator, only_available_clouds):
            self.__cloud = cloud
            self.__neighborhood = cloudNeighborhood.KnnCloudNeighborhood(cloud,num_neighbors)
            self.reward_aggregator = reward_aggregator
            self.__only_available_clouds = only_available_clouds

        def give_reward(self, reward):
            self.reward_aggregator.addReward(reward)

        def compute_distance(self, base_station, cloud):
            """
            computes the euclidian distance between a cloud and a base station
            :param base_station:
            :param cloud:
            :return:
            """
            userX, userY = base_station.get_pos()
            cloudX, cloudY = cloud.get_node().get_pos()
            return (userX-cloudX)**2 + (userY-cloudY)**2

        def closestCloud(self, service):
            closest_cloud = None
            closest_cloud_distance_sq = math.inf
            for cloud in self.__neighborhood.get_neighboring_clouds():
                if not self.__only_available_clouds or (self.__only_available_clouds and (cloud.totalMemoryRequirement()+service.get_memory_requirement()) / cloud.memory_capacity() <=1):
                    distance = self.compute_distance(service.get_user().get_base_station(), cloud)
                    if distance < closest_cloud_distance_sq:
                        closest_cloud_distance_sq = distance
                        closest_cloud = cloud
            return closest_cloud


        def leastOccupiedCloud(self, service):
            userX, userY = service.get_user().get_basestation().get_pos()
            least_occupied_cloud = None
            least_occupied_cloud_cloud_utilization = math.inf
            for cloud in self.__neighborhood.get_neighboring_clouds():
                utilization = cloud.totalMemoryRequirement()/cloud.memory_capacity()
                if utilization < least_occupied_cloud_cloud_utilization:
                    least_occupied_cloud_cloud_utilization = utilization
                    least_occupied_cloud = cloud
            return least_occupied_cloud

        def process_migration_event(self, service):
            migration_actions = []
            if service.lastUserNode is not service.user.get_base_station():
                target_cloud = self.closestCloud(service)
                if target_cloud:
                    return MigrationAction(service, self.__cloud, target_cloud)
            return NoMigrationAction(service, self.__cloud)

        def get_neighbor_clouds(self):
            return self.__neighborhood.get_neighboring_clouds()

    class InstanceHops(InstanceEuclidian):
        def __init__(self, cloud, num_neighbors, reward_aggregator, only_available_clouds):
            super(AlwaysMigrateToClosestInNeighborhoodAlgorithm.InstanceHops, self).__init__(cloud, num_neighbors, reward_aggregator, only_available_clouds)

        def compute_distance(self, base_station, cloud):
            """
            computes the number of hops between a cloud and a base station
            :param base_station:
            :param cloud:
            :return:
            """
            return base_station.num_hops_to(cloud.get_node())


class AlwaysMigrateToClosestAvailableInNeighborhoodAlgorithm(MigrationAlgorithm):
    """
    Always migrates services to the cloud that has the lowest latency to its user (Assumes full connectivity!)
    Note that this will only work with the simulation since it requires global knowledge about the entire topology.
    TODO: How about a simple algorithm that learns from experience which cloud has the lowest latency to which base station using a globally synchronized lookup table (aka. multi-armed bandit with epsilon-greedy updates)
    """
    def __init__(self, num_neighbors):
        self.__num_neighbors = num_neighbors

    def get_name(self):
        return "always_migrate_local_available"

    def create_instance(self, cloud):
        return AlwaysMigrateToClosestAvailableInNeighborhoodAlgorithm.Instance(cloud, self.__num_neighbors)

    class Instance(MigrationAlgorithm.Instance):
        def __init__(self, cloud, num_neighbors):
            self.__cloud = cloud
            self.__neighborhood = cloudNeighborhood.KnnCloudNeighborhood(cloud,num_neighbors)

        def euclidianClosestAvailableCloud(self, service, predicted_cloud_memory_requirement):
            userX, userY = service.get_user().get_base_station().get_pos()
            closest_cloud = None
            closest_cloud_distance_sq = math.inf
            for cloud in [self.__cloud]+self.__neighborhood.get_neighboring_clouds():
                cloudX, cloudY = cloud.get_node().get_pos()
                distance_sq = (userX-cloudX)**2 + (userY-cloudY)**2
                if not cloud in predicted_cloud_memory_requirement:
                    predicted_cloud_memory_requirement[cloud] = cloud.totalMemoryRequirement()
                predicted_utilization = (predicted_cloud_memory_requirement[cloud]+service.get_memory_requirement()) / cloud.memory_capacity()
                if predicted_utilization < 1 and distance_sq < closest_cloud_distance_sq:
                    closest_cloud_distance_sq = distance_sq
                    closest_cloud = cloud
                    predicted_cloud_memory_requirement[cloud] += service.get_memory_requirement()
            return closest_cloud


        def process_migration_event(self, sim):
            migration_actions = []
            for service in self.__cloud.get_services():
                if service.lastUserNode is not service.user.get_base_station():
                    predicted_cloud_utilizations = {}
                    target_cloud = self.euclidianClosestAvailableCloud(service, predicted_cloud_utilizations)
                    if target_cloud and not target_cloud == self.__cloud:
                        migration_actions.append(MigrationAction(service, target_cloud))
            return MigrationAlgorithm.MigrationResult(migration_actions)

        def get_neighbor_clouds(self):
            return self.__neighborhood.get_neighboring_clouds()