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


class MigrationAction:
    """
    Represents a migration action
    """

    def __init__(self, service, source_cloud, target_cloud):
        self.__service = service
        self.__target_cloud = target_cloud
        self.__source_cloud = source_cloud
        self.__failed = False

    def get_service(self):
        return self.__service

    def get_target_cloud(self):
        return self.__target_cloud

    def fail(self):
        """
        Call this function to mark this migration action as failed.
        :return: None
        """
        self.__failed = True

    def failed(self):
        """
        Returns if this migration action failed
        :return: True, if this migration was marked as failed, False if not
        """
        return self.__failed


class NoMigrationAction:
    """
    Represents a no-migration action (stay at cloud)
    """

    def __init__(self, service, cloud):
        self.__service = service
        self.__cloud = cloud
        assert self.__service.get_cloud() is self.__cloud

    def get_service(self):
        return self.__service

    def get_cloud(self):
        return self.__cloud


class ServiceCostFunction:
    """
    Abstract base class for per-service cost functions.
    """

    def calculate_cost(self, action):
        """
        Calculates the cost for either a MigrationAction or a NoMigratonAction
        :param action:
        :return:
        """
        if action is MigrationAction:
            self.calculate_migration_cost(action.get_service(), action.get_target_cloud())
        elif action is NoMigrationAction:
            self.calculate_no_migration_cost(action.get_service())
        else:
            raise ValueError("Expected a MigrationAction or NoMigrationAction.")

    @abc.abstractmethod
    def calculate_migration_cost(self, service, target_cloud):
        """
        Implement this method to calculate the cost of a migration action.
        Remember to check, if it failed.
        :param service: Service that was migrated
        :param target_cloud: Cloud it was migrated to (if it was successful)
        :return: A number that represents the cost.
        """
        pass

    @abc.abstractmethod
    def calculate_no_migration_cost(self, service):
        """
        Implement this method to calculate the cost of a no-migration action.
        :param service: the service that was not migrated
        :param cloud: the cloud where service remained
        :return: A number that represents the cost.
        """
        pass


class MigrationAlgorithm:

    class Instance:

        @abc.abstractmethod
        def process_migration_event(self, service):
            """
            determines migration actions for a cloud at a migration event
            :param cloud: the cloud that this function was invoked for
            :return: MigrationEvent or NoMigrationEvent
            """
            pass

        def give_reward(self, reward):
            """
            This function gives an immediate reward for the last action. If this method isn't called, the reward for the last action is 0.
            :param reward:
            :return: None
            """
            pass

    @abc.abstractmethod
    def create_instance(self, cloud):
        """
        This factory function creates an instance of a migration algorithm for a specific cloud.
        :param cloud: cloud of the created algorithm instance
        :return: an Algorithm Instance object
        """

    @abc.abstractmethod
    def get_name(self):
        return 'unnamed migration algorithm'