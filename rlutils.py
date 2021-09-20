import time
import numpy as np
from tqdm import tqdm
import argparse
import torch
import os


def solve_agent(agent, instances, config, evaluate_func, keyname):
    
    # modify instances based on size difference
    if config.diff_type == 'eq':
        _instances = instances.clone()
    if config.diff_type == 'small':
        _instances = make_big(instances, config.size_diff)
    agent_solution = {'solution': [], keyname: [], 'time': []}
    # configuring active search
    print_fl = False
    if not config.active_search:
        agent.actor.eval()
        agent.critic.eval()
    else:
        config.print_progress = False
        print_fl = True
    # solver main loop for each test case
    for i in tqdm(range(_instances.shape[0])):
        t = time.time()
        # creating local search batch
        test_data = _instances[i].repeat(config.batch_size, 1, 1)
        if config.active_search:
            agent.load_weights()
        # initializing solution
        best_value = -1e8 if config.maximize else 1e8
        best_sol = None
        # iteration loop
        for j in range(config.num_it):
            # solving by RL model
            if config.active_search:
                agent.learn(test_data)
                sol = agent.actor.indices.detach().cpu()
            else:
                sol = agent.solve_agent(test_data).detach().cpu()
            # taking best solution from the batch
            value = np.array(evaluate_func(instances[i].repeat(config.batch_size, 1, 1), sol))
            solidx = value.argmax() if config.maximize else value.argmin()
            if (value[solidx] > best_value and config.maximize) or (value[solidx] < best_value and not config.maximize):
                best_value = value[solidx]
                best_sol = sol[solidx].numpy()
        t = time.time() - t
        # appending solution and related measures
        agent_solution['solution'].append(best_sol)
        agent_solution[keyname].append(best_value)
        agent_solution['time'].append(t)
    # reverting print_progress param
    if print_fl: config.print_progress = True
    return agent_solution


def inference_config(config):
    
    loadpath = config.loadpath.replace('/size_%d'%config.problem_size, '')
    paths = os.listdir(loadpath)
    if len(paths) > 0:
        sizes = []
        for f in paths:
            try:
                s = int(f.replace('size_', ''))
            except:
                continue
            path = os.path.join(loadpath, f)
            if os.path.isdir(path) and 'size_' in f and len(os.listdir(path)):
                sizes.append(s)
        
        sizes = np.array(sorted(sizes))
        if sizes.shape[0] > 0:
            if config.problem_size in sizes:
                config.size_diff = 0
                config.diff_type = 'eq'
                return config
            else:
                valids = sizes[sizes > config.problem_size]
                if valids.shape[0] > 0:
                    config.diff_type = 'small'
                    config.size_diff = valids[0] - config.problem_size
                    config.problem_size = valids[0]
                    config.savepath = '_'.join(config.savepath.split('_')[:-1]) + '_%d'%config.problem_size
                    config.loadpath = '_'.join(config.loadpath.split('_')[:-1]) + '_%d'%config.problem_size
                else:
                    config.diff_type = 'big'
                    config.size_diff = sizes[-1] - config.problem_size
                    config.problem_size = sizes[-1]   ### not correct
                return config
    else:
        raise ValueError('%s folder is empty'%config.loadpath)


def make_big(x, size_inc):
    _x = []
    for i in range(x.shape[0]):
        _x.append(torch.cat((x[i], x[i, 0].repeat(size_inc, 1)), 0))
    return torch.stack(_x)