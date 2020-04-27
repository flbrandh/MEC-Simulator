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

import abc,math

class CloudNeighborhood:
    @abc.abstractmethod
    def get_neighboring_clouds(self):
        pass


class KnnCloudNeighborhood(CloudNeighborhood):
    def __init__(self, cloud, k):
        """
        Inits the knn by gathering the k nearest neighbors
        :param cloud:
        :param k:
        """
        cloud_node = cloud.get_node()
        self.__neighboring_clouds = cloud.get_nearest_clouds(k)

    def get_neighboring_clouds(self):
        return self.__neighboring_clouds;


class AllConnectedCloudNeighborhood(CloudNeighborhood):
    def __init__(self, cloud):
        """
        Inits the neighborhood by gathering all connected neighbors
        :param cloud:
        :param k:
        """
        cloud_node = cloud.get_node()
        self.__neighboring_clouds = cloud.get_nearest_clouds(math.inf)

    def get_neighboring_clouds(self):
        return self.__neighboring_clouds;
