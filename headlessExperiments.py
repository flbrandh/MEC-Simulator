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

import os, sys
import errno
import simulator
import Learning
import random
import basicCostFunctions
import json
import datetime
import basicMigrationAlgorithms

simscale = 2
num_simulation_steps = 1000000


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()

def run_experiment(id, numUsers, numEpochs, seed, target_model_update_frequency, network_depth, service_cost_function, migration_algorithm):
    rng = random.Random()
    rng.seed(seed)

    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = "simulation_output/sim_target_and_depth_test_" + date_time + "_" + str(
        id) + ".txt"
    hparam = Learning.DQNMigrationAlgorithm.Hyperparameters()
    hparam.batch_fraction_of_replay_memory = 1
    hparam.max_replay_memory_size = 1000
    hparam.episode_length = 1000
    hparam.num_epochs = 50
    hparam.network_depth = network_depth
    hparam.network_width = 20
    hparam.max_num_services = 3
    hparam.initial_exploration_boost = 0
    hparam.target_model_update_frequency = target_model_update_frequency

    #migration_algorithm = Learning.DQNMigrationAlgorithm(hparam, rng,True)
    #migration_algorithm = Learning.DQNMigrationAlgorithm(hparam, rng)

    simulation_config = simulator.Simulation.Configuration()
    simulation_config.numClouds = 15 * simscale
    simulation_config.numUsers = numUsers
    simulation_config.numServices = numUsers
    simulation_config.numInternalNodes = 15 * simscale
    simulation_config.numBaseStations = 10 * simscale
    simulation_config.migration_algorithm = migration_algorithm
    simulation_config.service_cost_function = service_cost_function #basicCostFunctions.ComplexCostFunctionNoActivation()
    simulation_config.movement_model = 'linear'

    sim = simulator.Simulation(simulation_config,rng)
    progress(0,num_simulation_steps)

    #simulate
    for i in range(num_simulation_steps):
        sim.step()
        if i % 500 == 0:
            progress(i,num_simulation_steps)

            # converte data to JSON
            data = {}
            data['num_simulation_steps'] = i
            data['random_seed'] = seed
            data['statistics'] = sim.statistics.getSerializableDict()
            data['avgRewards'] = migration_algorithm.shared_agent.avgRewards
            data['configuration'] = simulation_config.get_serializable_dict()
            data['hyperparameters'] = hparam.get_serializable_dict()
            if hasattr(migration_algorithm, 'shared_agent'):
                if hasattr(migration_algorithm.shared_agent,
                           'serialize_model'):
                    data['model_weights'] = [x.tolist() for x in
                                             migration_algorithm.shared_agent.serialize_model()]
            json_str = json.dumps(data, indent=4)

            # create directory if not existent
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise

            # write json
            f = open(filename, "w+")
            f.write(json_str)
            f.close()
    progress(num_simulation_steps, num_simulation_steps)


# run the experiment with multiple users:
threads = []

id = 0
for migration_algorithm in [basicMigrationAlgorithms.AlwaysMigrateToClosestInNeighborhoodAlgorithm(10, True, distance_measure='hops'), basicMigrationAlgorithms.AlwaysMigrateToClosestInNeighborhoodAlgorithm(100, True, distance_measure='hops'), basicMigrationAlgorithms.NeverMigrateAlgorithm()]:
    for service_cost_function in [basicCostFunctions.ComplexCostFunctionNoActivation(), basicCostFunctions.LatencyCostFunction()]:
        seed = 42
        if len(sys.argv) == 1 or (len(sys.argv) > 1 and int(sys.argv[1]) == id):
            run_experiment(id, simscale*20, 50, seed, 10, 3, service_cost_function, migration_algorithm)
        id += 1



