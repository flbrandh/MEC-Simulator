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

import simulator
import basicMigrationAlgorithms
from simplot import SimPlot, SimPlotLearning, SimPlotStats,SimPlotPygame
import Learning
import random
import basicCostFunctions
import os, json, datetime
import numpy as np

from tensorflow.keras import backend as K
import tensorflow as tf

config = tf.ConfigProto(log_device_placement=True,intra_op_parallelism_threads = 8, inter_op_parallelism_threads = 1)
session = tf.Session(config=config)
K.set_session(session)
rng = random.Random()
seed = 42
rng.seed(seed)
num_simulation_steps = 1000000

date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = "simulation_output/sim_gui_" + date_time + "_" + str(seed) + ".txt"
hparam=None

print("Choose the migration algorithm")
print("   1) never migrate")
print("   2) always migrate to closest globally")
print("   3) always migrate to closest available globally")
print("   4) always migrate to closest (euclidian) cloud in Neighborhood ")
print("   5) always migrate to closest (hops) cloud in Neighborhood ")
print("   6) always migrate to closest (hops) available cloud in Neighborhood ")
print("   7) shared DQN")
print("   8) shared DQN, model evaluation")
try:
    algorithm_choice_str = input("choice: ")
    algorithm_choice = int(algorithm_choice_str)
except ValueError:
    print("invalid option:'",algorithm_choice_str,"'")
    exit(-1)
if   algorithm_choice == 1:
    migration_algorithm = basicMigrationAlgorithms.NeverMigrateAlgorithm()
elif algorithm_choice == 2:
    migration_algorithm = basicMigrationAlgorithms.AlwaysMigrateAlgorithm(False)
elif algorithm_choice == 3:
    migration_algorithm = basicMigrationAlgorithms.AlwaysMigrateAlgorithm(True)
elif algorithm_choice == 4:
    migration_algorithm = basicMigrationAlgorithms.AlwaysMigrateToClosestInNeighborhoodAlgorithm(10, False, distance_measure = 'euclidian')
elif algorithm_choice == 5:
    migration_algorithm = basicMigrationAlgorithms.AlwaysMigrateToClosestInNeighborhoodAlgorithm(10, False, distance_measure='hops')
elif algorithm_choice == 6:
    migration_algorithm = basicMigrationAlgorithms.AlwaysMigrateToClosestInNeighborhoodAlgorithm(10, True, distance_measure='hops')
elif algorithm_choice == 7 or algorithm_choice == 8:
    hparam = Learning.DQNMigrationAlgorithm.Hyperparameters()
    hparam.batch_fraction_of_replay_memory = 1
    hparam.max_replay_memory_size = 1000
    hparam.episode_length = 1000
    hparam.num_epochs = 50
    hparam.network_depth = 3
    hparam.network_width = 20
    hparam.max_num_services = 3
    hparam.initial_exploration_boost = 0
    hparam.target_model_update_frequency = 10
    if algorithm_choice == 8:
        #disable exploration
        hparam.initial_exploration_boost = 0
        hparam.epsilon = 0
        #disable learning
        hparam.do_training = False
    migration_algorithm = Learning.DQNMigrationAlgorithm(hparam,rng,True)
else:
    print("invalid option:'", algorithm_choice_str, "'")
    exit(-1)

simscale = 2

simulation_config                       = simulator.Simulation.Configuration()
simulation_config.numClouds             = 15*simscale
simulation_config.numUsers              = 20*simscale
simulation_config.numServices           = 20*simscale#25*simscale
simulation_config.numInternalNodes      = 15*simscale
simulation_config.numBaseStations       = 10*simscale
simulation_config.migration_algorithm   = migration_algorithm
simulation_config.service_cost_function = basicCostFunctions.ComplexCostFunctionNoActivation()#ComplexCostFunction()
simulation_config.movement_model        ='linear'

sim = simulator.Simulation(simulation_config, rng)

if algorithm_choice == 8:
    log_file_name = input("log(model) filename: ")
    with open(log_file_name) as log_file:
        data = json.load(log_file)
        migration_algorithm.shared_agent.deserialize_model([np.asarray(mat) for mat in data['model_weights']])

#global migration baseline strategies might need the entire simulation
if hasattr(migration_algorithm, 'set_simulation'):
    migration_algorithm.set_simulation(sim)

enable_visualization = True
if enable_visualization:
    simPltPygame = SimPlotPygame(sim)

if hasattr(migration_algorithm, 'shared_agent'):
    simPltStats = SimPlotLearning(sim, migration_algorithm.shared_agent)
else:
    simPltStats = SimPlotStats(sim)

for i in range(num_simulation_steps):
    sim.step()
    if enable_visualization:
        simPltPygame.plot()
    if i % 500 is 0:
        print("simulation "+str(0.1*int(1000*(i/num_simulation_steps)))+"%")
        simPltStats.plot()

        # converte data to JSON
        data = {}
        data['num_simulation_steps'] = i
        data['random_seed'] = seed
        data['statistics'] = sim.statistics.getSerializableDict()
        data['avgRewards'] = migration_algorithm.shared_agent.avgRewards
        data['configuration'] = simulation_config.get_serializable_dict()
        if hparam:
            data['hyperparameters'] = hparam.get_serializable_dict()
        if hasattr(migration_algorithm, 'shared_agent'):
            if hasattr(migration_algorithm.shared_agent, 'serialize_model'):
                data['model_weights'] = [x.tolist() for x in migration_algorithm.shared_agent.serialize_model()]
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

