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

import abc

class CostFunctionInterface:
    """
    TODO this needs to be an interface for the cost in a neighborhood, which can be generalized to include the entir envionment of course. therefore define an interface for neighborhoods
    """
    @abc.abstractmethod
    def getCost(self, simulation):
        """
        returns the current total cost of an entire simulation
        :param simulation: SimulationInterface that provides access to the current state of the simulation
        :return: a number that represents an objective measurement of the current cost
        """
        pass


class CloudInterface:
    """
    Abstract interface for a user in an edge cloud system
    """
    @abc.abstractmethod
    def get_total_available_memory(self):
        pass

    @abc.abstractmethod
    def get_total_required_memory(self):
        pass

    @abc.abstractmethod
    def get_services(self):
        pass

    @abc.abstractmethod
    def get_nearest_clouds(self,k):
        """
        Determines the k nearest clouds.
        :param k: number of desired clouds
        :return: array of length <= k, containing the nearest clouds in the ascending order of their distance
        """
        pass

class ServiceInterface:
    """
    Abstract interface for a service in an edge cloud system.
    """
    @abc.abstractmethod
    def get_cloud(self):
        """
        returns the cloud of this node
        :return:
        """
        pass

    @abc.abstractmethod
    def get_user(self):
        """
        returns the usr of this service
        :return: UserInterface of the user that belongs to th service
        """
        pass

    @abc.abstractmethod
    def get_memory_requirement(self):
        """
        Returns the memory requirement of the service
        :return: the memory requirement in bytes
        """
        pass

    @abc.abstractmethod
    def get_latency_requirement(self):
        """
        Returns the latency requirement of the service.
        :return: required latency by this service in ms.
        """
        pass


class EdgeCloudSystemInterface:
    """
    Abstract interface for an edge cloud system
    """

    @abc.abstractmethod
    def get_clouds(self):
        """
        returns all clouds of the simulation
        :return: list of CloudInterface with all clouds of the simulation
        """
        pass

    @abc.abstractmethod
    def get_users(self):
        """
        returns all users of the simulation
        :return: list of UserInterface
        """
        pass