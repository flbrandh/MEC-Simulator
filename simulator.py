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

import random
import math
import itertools
from simulatorInterface import *
import migrationAlgorithmInterface


class Node:
    def __init__(self,  pos):
        self.neighbors = []
        self.pos = pos
        self.cloud = None

    def get_cloud(self):
        return self.cloud

    def setCloud(self, cloud):
        self.cloud = cloud

    def addNeighbor(self, neighbor):
        if not neighbor in self.neighbors:
            self.neighbors.append(neighbor)
            neighbor.addNeighbor(self)

    def getNeighbors(self):
        return self.neighbors

    def get_pos(self):
        return self.pos

    def num_hops_to(self, targetNode):
        #dijkstra's algorithm:
        fringe = []
        closed = []
        shortestDistances = {}

        fringe.append(self)
        shortestDistances[self] = 0

        while fringe:
            nextNode = None
            lowest_dist_in_fringe = math.inf
            # find closest node in fringe
            for node in fringe:
                d = shortestDistances[node]
                if d < lowest_dist_in_fringe:
                    lowest_dist_in_fringe = d
                    nextNode = node
            # remove the cosest node from the fringe and add it to the closed nodes
            fringe.remove(nextNode)
            distToNextNode = shortestDistances[nextNode]
            if nextNode is targetNode:
                return distToNextNode
            closed.append(nextNode)
            for neighbor in nextNode.getNeighbors():
                if not neighbor in closed:
                    if not neighbor in shortestDistances:
                        shortestDistances[neighbor] = distToNextNode+1
                    elif distToNextNode+1 < shortestDistances[neighbor] :
                        shortestDistances[neighbor] = distToNextNode+1
                    if neighbor not in fringe:
                        fringe.append(neighbor)
        return math.inf


class BaseStation(Node):
    def __init__(self, pos):
        super(BaseStation, self).__init__(pos)


class BrownianUser:
    def __init__(self, pos, basestation,rng):
        self.rng = rng
        self.pos = pos
        self.basestation = basestation
        self.services = []
        self.speed = 0.1#random.uniform(0.005,0.01)#0.02

    def move(self):
        # brownian movement on a donut world
        self.pos = ((self.pos[0] + self.rng.uniform(-self.speed,self.speed))%1.0, (self.pos[1] + self.rng.uniform(-self.speed,self.speed))%1.0)

    def get_pos(self):
        return self.pos

    def assignBaseStation(self, basestation):
        self.basestation = basestation

    def get_base_station(self):
        return self.basestation

    def getServices(self):
        return self.services

class LinearUser(BrownianUser):
    def __init__(self, pos, basestation,rng):
        super(LinearUser,self).__init__(pos,basestation,rng)
        self.destination = self.__new_destination()
        self.speed = 0.025

    def __new_destination(self):
        return (self.rng.random(), self.rng.random())

    def move(self):
        x, y = self.get_pos()
        dx, dy = self.destination[0]-x, self.destination[1]-y
        distance = math.sqrt(dx**2+dy**2)
        if distance < self.speed:
            self.pos = self.destination
            self.destination = self.__new_destination()
        else:
            dx, dy = dx/distance, dy/distance
            self.pos = x + self.speed * dx, y + self.speed * dy


class Service(ServiceInterface):
    def __init__(self, user, memory_requirement, latency_requirement):
        self.memoryRequirement = memory_requirement#1#random.uniform(0.1,1)
        self.latencyRequirement = latency_requirement#4#random.uniform(2,5) #hops
        self.__current_cloud = None
        self.__last_cloud = None
        self.active = False
        self.user = user
        self.user.services.append(self)

        #a little optimization (caching the latency):
        self.lastServiceNode = None
        self.lastUserNode = None
        self.lastNumHops = None

        self.last_migration_event_step = 0

    def __del__(self):
        self.user.services.remove(self)

    def measuredLatency(self):
        serviceNode = self.__current_cloud.get_node()
        userNode = self.user.get_base_station()
        if serviceNode is self.lastServiceNode and userNode is self.lastUserNode:
            numHops = self.lastNumHops
        else:
            numHops = serviceNode.num_hops_to(
                userNode) + 1  # +1 is the radio link hop
        self.lastNumHops = numHops
        self.lastUserNode = userNode
        self.lastServiceNode = serviceNode
        return numHops

    def latencyRequirementFulfilled(self):

        return self.measuredLatency() < self.latencyRequirement;

    def get_user(self):
        return self.user

    def get_memory_requirement(self):
        return self.memoryRequirement

    def get_latency_requirement(self):
        return self.latencyRequirement

    def set_cloud(self, cloud):
        self.__last_cloud = self.__current_cloud
        self.__current_cloud = cloud

    def get_cloud(self):
        return self.__current_cloud

    def get_last_cloud(self):
        return self.__last_cloud

    def is_active(self):
        return self.active

class Cloud:

    def __init__(self, node, memory_capacity):
        self.node = node
        self.memoryCapacity = memory_capacity
        self.services = []
        self.__migration_algorithm_instance = None


    def get_pos(self):
        return self.node.get_pos()

    def addService(self, service):
        if service in self.services:
            return # do nothing
        if self.totalMemoryRequirement() + service.memoryRequirement <= self.memoryCapacity:
            service.active = True
        else:
            assert service.active == False
            service.active = False

        if service.get_cloud():
            service.get_cloud().removeService(service)
        service.set_cloud(self)
        self.services.append(service)

    def removeService(self, service):
        numServicesBefore = len(self.services)
        self.services.remove(service)
        # determine which services can be active after the removal
        memoryUtilization = 0
        for service in self.services:
            if memoryUtilization + service.memoryRequirement <= self.memoryCapacity:
                service.active = True
                memoryUtilization += service.memoryRequirement
            else:
                service.active = False
        assert numServicesBefore -1 == len(self.services)


    def totalMemoryUtilization(self):
        """Returns the memory consumption of all active services"""
        memoryUtilization = 0
        for service in self.services:
            if service.active:
                memoryUtilization += service.memoryRequirement
        return memoryUtilization

    def totalMemoryRequirement(self):
        memoryRequirement = 0
        for service in self.services:
            memoryRequirement += service.memoryRequirement
        return memoryRequirement

    def memory_capacity(self):
        return self.memoryCapacity

    def get_node(self):
        return self.node

    def get_services(self):
        return self.services

    def set_migration_algorithm_instance(self,mai):
        self.__migration_algorithm_instance = mai

    def get_migration_algorithm_instance(self):
        return self.__migration_algorithm_instance

    def get_nearest_clouds(self, k):
        self_node = self.get_node()
        knn_clouds = []
        #djikstra's algorithm until knn_clouds are found
        fringe = []
        closed = []
        shortestDistances = {}

        fringe.append(self_node)
        shortestDistances[self_node] = 0

        while fringe:
            nextNode = None
            lowest_dist_in_fringe = math.inf
            #find closest node in fringe
            for node in fringe:
                d = shortestDistances[node]
                if d < lowest_dist_in_fringe:
                    lowest_dist_in_fringe = d
                    nextNode = node
            #remove the cosest node from the fringe and add it to the closed nodes
            fringe.remove(nextNode)
            distToNextNode = shortestDistances[nextNode]
            next_node_cloud = nextNode.get_cloud()
            #collect cloud and rturn if enough clouds are collected
            if next_node_cloud and next_node_cloud is not self:
                knn_clouds.append(next_node_cloud)
                if len(knn_clouds) is k:
                    assert not self in knn_clouds
                    return knn_clouds
            closed.append(nextNode)
            for neighbor in nextNode.getNeighbors():
                if not neighbor in closed:
                    if not neighbor in shortestDistances:
                        shortestDistances[
                            neighbor] = distToNextNode + 1
                    elif distToNextNode + 1 < shortestDistances[neighbor]:
                        shortestDistances[
                            neighbor] = distToNextNode + 1
                    if neighbor not in fringe:
                        fringe.append(neighbor)
        assert self not in knn_clouds
        return knn_clouds


class CloudState:
    """Abstract Base Class that provides the state of a cloud"""
    def asVector(self):
        return []

    @abc.abstractmethod
    def getLocation(self):
        pass

    @abc.abstractmethod
    def getServiceStates(self):
        pass


class ThinCloudState(CloudState):
    def __init__(self, cloud):
        self.cloud = cloud

    def getServiceStates(self):
        for service in self.cloud.get_services():
            yield ThinServiceState(service)

class ServiceState:
    """Abstract Base Class that offers the state of a service"""
    def asVector(self):
        return [list(self.getUserLocation()),
                list(self.getMemoryRequirement()),
                list(self.getLatencyRequirement())]

    @abc.abstractmethod
    def getUserLocation(self):
        pass

    @abc.abstractmethod
    def getMemoryRequirement(self):
        pass

    @abc.abstractmethod
    def getLatencyRequirement(self):
        pass

class ThinServiceState(ServiceState):
    def __init__(self, service):
        self.service = service

    def getUserLocation(self):
        return self.service.get_user().get_pos()

    def getMemoryReqirement(self):
        return self.service.get_memory_requirement()

    def getLatencyRequirement(self):
        return self.service.get_latency_requirement()


class Network:
    def __init__(self, nodes, clouds):
        self.__nodes = nodes
        self.__clouds = clouds

    def from_serialized_dict(self,parsed_json):
        # TODO implement
        pass

    def to_serializable_dict(self):
        #TODO implemenent
        pass

    def nodes(self):
        return self.nodes()

    def clouds(self):
        return self.clouds()


class Simulation(EdgeCloudSystemInterface):

    class Configuration:
        def __init__(self):
            self.numClouds = 0
            self.numUsers = 0
            self.numServices = 0
            self.assign_services_randomly = False
            self.numInternalNodes = 0
            self.numBaseStations = 0
            self.cloud_memory_capacity = 3
            self.service_memory_requirement = 1
            self.service_latency_requirement = 5
            self.migration_algorithm = None
            self.service_cost_function = None
            self.movement_model = 'brownian'

        def get_serializable_dict(self):
            dict = {}
            dict['numClouds']                   = self.numClouds
            dict['numUsers']                    = self.numUsers
            dict['numServices']                 = self.numServices
            dict['assign_services_randomly']    = self.assign_services_randomly
            dict['numInternalNodes']            = self.numInternalNodes
            dict['numBaseStations']             = self.numBaseStations
            dict['cloud_memory_capacity']       = self.cloud_memory_capacity
            dict['service_memory_requirement']  = self.service_memory_requirement
            dict['service_latency_requirement'] = self.service_latency_requirement
            dict['migration_algorithm']         = self.migration_algorithm.get_name()
            dict['service_cost_function']       = self.service_cost_function.get_name()
            dict['movement_model']              = self.movement_model
            return dict

    class Statistics:
        def __init__(self, sim):
            self.sim = sim
            # self.cost = []
            self.dissatisfactionRate = []
            self.inactiveRate = []
            self.num_migrations = []
            self.num_migration_events = []
            self.num_proposed_migrations = []
            self.avg_migration_dist_to_user_bs = []
            self.avg_latency = []

        def getSerializableDict(self):
            dict = {}
            dict['num_migrations'] = self.num_migrations
            dict['num_migration_events'] = self.num_migration_events
            dict['num_proposed_migrations'] = self.num_proposed_migrations
            dict['avg_latency'] = self.avg_latency
            return dict

        def getNumInactiveServices(self):
            numInactive = 0
            for service in self.sim.services:
                if not service.active:
                    numInactive += 1
            return numInactive

        def getNumDissatisfiedServices(self):
            numDissatisfied = 0
            for service in self.sim.services:
                if not service.latencyRequirementFulfilled():
                    numDissatisfied += 1
            return numDissatisfied

        def get_avg_latency(self):
            summed_latency = 0
            for service in self.sim.services:
                summed_latency += service.measuredLatency()
            return summed_latency/len(self.sim.services)

        def addSimulationStep(self):
            # self.cost.append(self.sim.get_cost())
            self.dissatisfactionRate.append(self.getNumDissatisfiedServices()/len(self.sim.services))
            self.inactiveRate.append(self.getNumInactiveServices()/len(self.sim.services))
            self.num_migrations.append(self.sim.current_step_num_migrations)
            self.num_proposed_migrations.append(self.sim.current_step_num_proposed_migrations)
            avg_migrated_dist = 0
            if self.sim.current_step_num_migrations > 0:
                avg_migrated_dist = self.sim.current_step_summed_migration_dist_to_user_bs / self.sim.current_step_num_migrations
            self.avg_migration_dist_to_user_bs.append(avg_migrated_dist)
            self.num_migration_events.append(self.sim.current_step_num_migration_events)
            self.avg_latency.append(self.get_avg_latency())

        def getNumSteps(self):
            return len(self.avg_latency)
            # return len(self.cost)


    def __init__(self, configuration, rng = random.Random()):
        self.rng = rng
        self.config = configuration
        self.statistics = Simulation.Statistics(self)

        self.baseStations = [BaseStation(pos=(rng.uniform(0,1),rng.uniform(0,1))) for x in range(self.config.numBaseStations)]
        self.nodes = self.baseStations + [Node(pos=(rng.uniform(0,1), rng.uniform(0,1))) for x in range(self.config.numInternalNodes)]
        self.connectNodesMST()
        self.connectNodesKNN(3)
        self.clouds = []
        #make sure that each node has at most one cloud
        for ci in range(min(self.config.numClouds,len(self.nodes))):
            while True:
                node = self.nodes[rng.randint(0,len(self.nodes)-1)]
                if not node.get_cloud():
                    cloud = Cloud(node = node,memory_capacity=self.config.cloud_memory_capacity)
                    self.clouds.append(cloud)
                    node.setCloud(cloud)
                    break
        # initialize the migration algorithms (needs to be none after the networ architecture is completely defined)
        for cloud in self.clouds:
            cloud.set_migration_algorithm_instance(
                self.config.migration_algorithm.create_instance(cloud))



        user_positions = [(rng.uniform(0,1),rng.uniform(0,1)) for x in range(self.config.numUsers)]
        self.users = []
        for pos in user_positions:
            new_user_basestation = self.getClosestBasestation(pos)
            new_user = None
            if self.config.movement_model == 'brownian':
                new_user = BrownianUser(pos=pos, basestation = self.getClosestBasestation(pos), rng = self.rng)
            elif self.config.movement_model == 'linear':
                new_user = LinearUser(pos=pos, basestation = self.getClosestBasestation(pos), rng = self.rng)
            else:
                raise ValueError("invalid movement model specified in parameter 'movment_model'. Valid otions are 'brownian' and 'linear'.")
            self.users.append(new_user)

        self.services = []
        for i, si in enumerate(range(self.config.numServices)):
            user = None
            if self.config.assign_services_randomly:
                user = self.users[rng.randint(0,self.config.numUsers-1)]
            else:
                user = self.users[i%len(self.users)]
            service = Service(user,self.config.service_memory_requirement, self.config.service_latency_requirement)
            self.services.append(service)
            self.clouds[rng.randint(0,len(self.clouds)-1)].addService(service) # random cloud
            #self.closestCloud(user.getBaseStation()).addService(service) # closest cloud

        totalMemoryCapacity = 0
        for cloud in self.clouds:
            totalMemoryCapacity += cloud.memoryCapacity
        totalMemoryRequirement = 0
        for service in self.services:
            totalMemoryRequirement += service.memoryRequirement
        print("ratio of total memory requirement to capacity: "+'{0:.2f}'.format((totalMemoryRequirement/totalMemoryCapacity)*100)+"%")

    def getStatistics(self):
        return self.statistics

    def getConfig(self):
        return self.config

    def get_nodes(self):
        return self.nodes

    def get_clouds(self):
        return self.clouds

    def get_users(self):
        return self.users

    def get_services(self):
        return self.services

    def getClosestBasestation(self,pos):
        closestDistance = math.inf
        closestBs = None
        for bs in self.baseStations:
            distance = (pos[0] - bs.get_pos()[0]) ** 2 + (pos[1] - bs.get_pos()[1]) ** 2
            if distance < closestDistance:
                closestDistance = distance
                closestBs = bs
        return closestBs

    def closestCloud(self, node, conditionFunction = lambda cloud : True):
        closestCloud = None
        closestCloudDistance = math.inf
        for cloud in self.clouds:
            if conditionFunction(cloud):
                dist = cloud.get_node().num_hops_to(node)
                if dist < closestCloudDistance:
                    closestCloud = cloud
                    closestCloudDistance = dist
        return closestCloud


    def connectNodesMST(self):
        freeNodes = self.nodes[1:]
        closedNodes = self.nodes[0:1]
        while freeNodes:
            shortestConnection = math.inf
            shortestConnectionClosedNode = None
            shortestConnectionFreeNode = None
            for closedNode, freeNode in itertools.product(closedNodes,freeNodes):
                distance = (closedNode.get_pos()[0] - freeNode.get_pos()[0]) ** 2 + (closedNode.get_pos()[1] - freeNode.get_pos()[1]) ** 2
                if distance < shortestConnection:
                    shortestConnection = distance
                    shortestConnectionClosedNode = closedNode
                    shortestConnectionFreeNode = freeNode
            shortestConnectionClosedNode.addNeighbor(shortestConnectionFreeNode)
            closedNodes.append(shortestConnectionFreeNode)
            freeNodes.remove(shortestConnectionFreeNode)

    def connectNodesKNN(self,N):
        for node in self.nodes:
            KNN = []
            for i in range(N+1):
                closest = None
                closestSqDist = math.inf
                for otherNode in self.nodes:
                    if (not otherNode is node) and (not otherNode in KNN):
                        sqDist = (node.get_pos()[0] - otherNode.get_pos()[0]) ** 2 + \
                                 (node.get_pos()[1] - otherNode.get_pos()[1]) ** 2
                        if sqDist < closestSqDist:
                            closestSqDist = sqDist
                            closest = otherNode
                KNN.append(closest)
                node.addNeighbor(closest)


    def getNeighboringClouds(self):
        return self.clouds

    def getCloudState(self, cloud):
        return cloud.get_pos(), cloud.memoryCapacity

    def getServiceState(self, service):
        return service.memoryRequirement, service.latencyRequirement,\
               self.getCloudState(service.currentCloud),\
               [self.getCloudState(neighborCloud) for neighborCloud in self.getNeighboringClouds()]

    # def get_cost(self):
    #     return -self.get_reward()

    def get_reward(self, action):
        return -self.config.service_cost_function.calculate_cost(action)


    def get_current_step(self):
        return self.statistics.getNumSteps()

    def step(self):
        # reset step-statistics
        self.current_step_num_migrations = 0
        self.current_step_num_proposed_migrations = 0
        self.current_step_summed_migration_dist_to_user_bs = 0.0
        self.current_step_num_migration_events = 0

        triggered_services = set()

        for user in self.users:
            user.move()
            new_closest_base_station = self.getClosestBasestation(user.get_pos())
            if(new_closest_base_station is not user.get_base_station()):
                user.assignBaseStation(new_closest_base_station)
                for service in user.getServices():
                    triggered_services.add(service)

        for service in self.services:
            if self.get_current_step() - service.last_migration_event_step > 5: # TODO: don't hard code the interval
                triggered_services.add(service)

        # now, process all migration events in random order:
        unique_triggered_clouds_in_random_order = list(triggered_services)
        random.shuffle(unique_triggered_clouds_in_random_order,self.rng.random)
        for service in unique_triggered_clouds_in_random_order:
            self.trigger_migration_event(service)
            service.last_migration_event_step = self.get_current_step()

        self.statistics.addSimulationStep()

    def trigger_migration_event(self, service):
        self.current_step_num_migration_events += 1
        cloud = service.get_cloud()
        cloud.last_migration_event_step = self.get_current_step()
        action = cloud.get_migration_algorithm_instance().process_migration_event(service)
        if isinstance(action, migrationAlgorithmInterface.MigrationAction):
            self.execute_migration_action(action)
        else:
            assert isinstance(action, migrationAlgorithmInterface.NoMigrationAction)
        cloud.get_migration_algorithm_instance().give_reward(self.get_reward(action))


    def execute_migration_action(self, action):
        self.current_step_num_proposed_migrations += 1
        service = action.get_service()
        cloud = action.get_target_cloud()
        sufficient_memory = service.get_memory_requirement() + cloud.totalMemoryRequirement() <= cloud.memory_capacity()
        if not sufficient_memory:
            action.fail()
        else:
            if not service.active or sufficient_memory:
                cloud.addService(action.get_service())
                self.current_step_num_migrations += 1
                pn = action.get_target_cloud().get_node().get_pos()
                pu = action.get_service().get_user().get_base_station().get_pos()
                dist_to_user_bs = math.sqrt(
                    (pn[0] - pu[0]) ** 2 + (pn[1] - pu[1]) ** 2)
                self.current_step_summed_migration_dist_to_user_bs += dist_to_user_bs
