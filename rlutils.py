import time
import numpy as np
from tqdm import tqdm


def solve_agent(agent, instances, config, evaluate_func, keyname):
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
    for i in tqdm(range(instances.shape[0])):
        t = time.time()
        # creating local search batch
        test_data = instances[i].repeat(config.batch_size, 1, 1)
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
                sol = agent.actor.indices
            else:
                sol = agent.solve_agent(test_data)
            # taking best solution from the batch
            value = np.array(evaluate_func(test_data, sol))
            solidx = value.argmax() if config.maximize else value.argmin()
            if (value[solidx] > best_value and config.maximize) or (value[solidx] < best_value and not config.maximize):
                best_value = value[solidx]
                best_sol = sol[solidx].detach().cpu().numpy()
        t = time.time() - t
        # appending solution and related measures
        agent_solution['solution'].append(best_sol)
        agent_solution[keyname].append(best_value)
        agent_solution['time'].append(t)
    # reverting print_progress param
    if print_fl: config.print_progress = True
    return agent_solution


