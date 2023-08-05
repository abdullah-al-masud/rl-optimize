import torch
import numpy as np
from msdlib import msd
import time
import pandas as pd
import seaborn as sns
import utils
import tsp_utils
from tsp_utils import Agent
from config import prepare_tsp_config
from rlutils import solve_agent, inference_config

sns.set()


def main(config):
    
    tsp_instances = tsp_utils.generate_tsp_instances(config, inference=True)
    print('tsp_instances.shape:', tsp_instances.shape)

    # solutions from all algorithms
    solutions = {}

    ## Ortools solution
    print('solving by ortools... ')
    or_solutions, or_time = tsp_utils.solve_tsp_ortools(tsp_instances.numpy())
    or_distances = tsp_utils.calculate_distances(tsp_instances.numpy(), or_solutions)
    solutions['ortools'] = {'solution': or_solutions, 'distance': or_distances, 'time': or_time}

    # RL model's training or inference
    print('\ncreating RL agent...')
    if config.inference:
        config = inference_config(config)
        agent = Agent(config)
        agent.load_weights()
        print('model weights are loaded for inference... ')
    else:
        config.size_diff = 0
        config.diff_type = 'eq'
        agent = Agent(config)
        print('training is being started...')
        t = time.time()
        for i in range(config.epoch):
            data = tsp_utils.generate_tsp_instances(config)
            agent.learn(data)
        agent.plot_learning()
        print('training is complete! elapsed time: ', pd.Timedelta(seconds=time.time()-t))

    # model performance evaluation
    print("\nevaluating the model's performance...")
    evaluate_func = tsp_utils.calculate_distances
    solutions['PN_model'] = solve_agent(agent, tsp_instances, config, evaluate_func, 'distance')

    ## Results
    comp_table = utils.comparison_table(solutions, keyname='distance', maximize=config.maximize)
    print(comp_table, '\n')

    utils.result_stats(solutions, 'distance', basename='ortools', other_keys=['time'])
    
    if config.show_sample:
        tsp_utils.plot_tsp_solutions(tsp_instances, solutions, 5)
        i = np.random.randint(config.num_test)
        print('sample solution of instance: %d'%i)
        print(np.array(solutions['ortools']['solution'][i]))
        print(np.array(solutions['PN_model']['solution'][i]))
    
    
if __name__ == '__main__':
    # getting config
    config = prepare_tsp_config()
    main(config)