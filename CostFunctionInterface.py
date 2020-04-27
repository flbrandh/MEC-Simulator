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
from migrationAlgorithmInterface import MigrationAction, NoMigrationAction

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
        if isinstance(action, MigrationAction):
            return self.calculate_migration_cost(action.get_service(), action.get_target_cloud(), action.failed())
        elif isinstance(action, NoMigrationAction):
            return self.calculate_no_migration_cost(action.get_service())
        else:
            raise ValueError("Expected a MigrationAction or NoMigrationAction.")

    @abc.abstractmethod
    def calculate_migration_cost(self, service, target_cloud, failed):
        """
        Implement this method to calculate the cost of a migration action.
        Remember to check, if it failed.
        :param service: Service that was migrated
        :param target_cloud: Cloud it was migrated to (if it was successful)
        :param failed: True, if the migration action failed, False, if it was successful
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

    @abc.abstractmethod
    def get_name(self):
        return 'unnamed cost function'