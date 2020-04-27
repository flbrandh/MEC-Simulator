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

import CostFunctionInterface
import math


class LatencyCostFunction(CostFunctionInterface.ServiceCostFunction):
    """
    Cost is based on measured latency of the service.
    """

    def calculate_migration_cost(self, service, target_cloud, failed):
        return service.measuredLatency()

    def calculate_no_migration_cost(self, service):
        return service.measuredLatency()

    def get_name(self):
        return 'LatencyBasedCostFunction'


class ProximityCostFunction(CostFunctionInterface.ServiceCostFunction):
    """
    Cost is based on measured latency of the service.
    """

    def distance(self, service):
        user_x, user_y = service.get_user().get_base_station().get_pos()
        cloud_x, cloud_y = service.get_cloud().get_pos()
        distance = math.sqrt(
            (user_x - cloud_x) ** 2 + (user_y - cloud_y) ** 2)
        return distance

    def calculate_migration_cost(self, service, target_cloud, failed):
        return self.distance(service)

    def calculate_no_migration_cost(self, service):
        return self.distance(service)

    def get_name(self):
        return 'ProximityCostFunction'


class ComplexCostFunction(CostFunctionInterface.ServiceCostFunction):
    """
    Cost is based on latency and incorporates migration overhead. Also incorporates an activation cost.
    """

    def cloud_activation_cost(self, service):
        cloud = service.get_cloud()
        single_cloud_activation_cost = 1 #equivalent to 1 hop of latency
        return single_cloud_activation_cost/len(cloud.get_services())

    def calculate_migration_cost(self, service, target_cloud, failed):
        migration_cost = 0
        if not failed:
            migration_cost = 1
        return service.measuredLatency() + migration_cost + self.cloud_activation_cost(service)

    def calculate_no_migration_cost(self, service):
        return service.measuredLatency() + self.cloud_activation_cost(service)

    def get_name(self):
        return 'ComplexCostFunction'


class ComplexCostFunctionNoActivation(CostFunctionInterface.ServiceCostFunction):
    """
    Cost is based on latency and incorporates migration overhead.
    """

    def calculate_migration_cost(self, service, target_cloud, failed):
        migration_cost = 0
        if not failed:
            migration_cost = 1
        return service.measuredLatency() + migration_cost

    def calculate_no_migration_cost(self, service):
        return service.measuredLatency()

    def get_name(self):
        return 'ComplexCostFunctionNoActivation'