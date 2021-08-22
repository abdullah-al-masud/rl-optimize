from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.spatial import distance_matrix
import numpy as np
import matplotlib.pyplot as plt
from msdlib import msd
import torch
import time
from tqdm import tqdm
import pandas as pd
from pointer_net import PN_Actor, PN_Critic


def create_data_model(input_data, start_index=0):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = input_data
    data['num_vehicles'] = 1
    data['depot'] = start_index
    return data


def get_solution(manager, routing, solution, start_index):
    """Prints solution on console."""
    index = routing.Start(start_index)
    solutions = []
    while not routing.IsEnd(index):
        solutions.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    solutions.append(manager.IndexToNode(index))
    return solutions


def solve_tsp(input_data, start_index=0):
    """Entry point of the program."""
    # Instantiate the data problem.
    dist_mat = distance_matrix(input_data, input_data)
    data = create_data_model(dist_mat, start_index)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return get_solution(manager, routing, solution, start_index)


def solve_tsp_ortools(_data):
    data = _data * 1000
    or_solutions = []
    runtime = []
    for i in tqdm(range(data.shape[0])):
        t = time.time()
        or_solutions.append(solve_tsp(data[i].tolist()))
        runtime.append(time.time() - t)
    return or_solutions, runtime


def calculate_distances(instances, solutions):
    return [np.sqrt(np.square(instances[i][solutions[i][1:]] - instances[i][solutions[i][:-1]]).sum(axis=1)).sum() for i in range(instances.shape[0])]


# solutions will be a dict containing <name>: {'solution': [], 'distance': [], 'time': []}
def plot_tsp_solutions(instances, solutions, num_plot=5, plot_index=[], title=''):

    n_algo = len(solutions)
    if title == '':
        title = 'TSP routes'
    if len(plot_index) == 0:
        indices = np.random.choice(
            instances.shape[0], size=num_plot, replace=False)
    else:
        indices = plot_index.copy()

    colors = msd.get_named_colors()
    for i in sorted(indices):
        fig_title = title + ' (index: %d)' % i
        fig, ax = plt.subplots(figsize=(5 * n_algo, 5), ncols=n_algo)
        if n_algo == 1:
            ax = [ax]
        fig.suptitle(fig_title, y=1.04, fontsize=12.5, fontweight='bold')
        for j, name in enumerate(solutions):
            ax[j].plot(instances[i][solutions[name]['solution'][i], 0], instances[i]
                       [solutions[name]['solution'][i], 1], color=colors[j], marker='o')
            ax[j].set_title('%s solution\ndistance: %.3f; time: %.4f' % (
                name, solutions[name]['distance'][i], solutions[name]['time'][i]))
        fig.tight_layout()
        plt.show()


def generate_tsp_instances(config, inference=False, num_test=None):
    if inference:
        if num_test is None:
            num_test = config.num_test
        data = torch.rand(num_test, config.problem_size, config.dimension)
    else:
        data = torch.rand(config.batch_size,
                          config.problem_size, config.dimension)
    return data


class Agent():
    
    def __init__(self, config, dtype=torch.float32):
        
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() and 'cuda' in config.device else "cpu")
        self.config = config
        
        # initializing models
        self.actor = PN_Actor(config).to(device=self.device)
        self.critic = PN_Critic(config).to(device=self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr)
        self.decay_actor = torch.optim.lr_scheduler.StepLR(self.opt_actor, step_size=self.config.lr_step, gamma=self.config.lr_step_decay)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr)
        self.decay_critic = torch.optim.lr_scheduler.StepLR(self.opt_critic, step_size=self.config.lr_step, gamma=self.config.lr_step_decay)
        
        # setting model mode
        if self.config.inference:
            self.actor.eval()
            self.critic.eval()
        else:
            if self.config.resume_training:
                self.load_weights()
            self.actor.train()
            self.critic.train()
        
        # initializing other values
        self.range_index = torch.tensor(list(range(self.config.batch_size)), device=self.device).long()
        self.reload_factor = .5 if self.config.maximize else 1.5
        self.ct_reward = -1e8 if self.config.maximize else 1e8
        self.push_factor = 1 + .02 if self.config.maximize else 1 - .02
        self.loss_direction = -1 if self.config.maximize else 1
        self.grad_clip = 1
        self.it = 0
        self.reward_stack = []
        self.others_stack = []
        
    def check_input(self, x):
        # formatting input
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(dtype=self.dtype, device=self.device)
        return x
    
    def solve_agent(self, x):
        x = self.check_input(x)
        # setting temperature
        self.actor.T = self.config.T
        # actor ouput
        out_index = self.actor(x)
        
        return out_index
    
    def learn(self, x):
        
        # setting temperature 1
        self.actor.T = 1
        # predicting from actor
        x = self.check_input(x)
        # print('x.shape:', x.shape, x.min())
        indices = self.actor(x)
        # print('incides.shape:', indices.shape, indices.min())
        log_prob = self.actor.log_prob
        # print('log_prob.shape:', log_prob.shape, log_prob.min())
        
        # reward
        reward = self.get_reward(x, indices).detach()
        # print('reward.shape:', reward.shape, reward.min())
        
        # # exponential baseline calculation
        # if self.it == 0:
        #     self.avg_baseline = reward
        # else:
        #     self.avg_baseline = reward * (1 - self.config.alpha) + self.avg_baseline * self.config.alpha
        # V = self.avg_baseline
        
        # baseline prediction from critic
        V = self.critic(x) #  / self.critic(x) * reward.min()
        # print('V.shape:', V.shape, V.min())
        
        # advantage = (reward - V)
        # print('advantage.shape:', advantage.shape, advantage.min())
        
        # loss
        actor_loss = self.loss_direction * ((reward - V).detach() * log_prob).mean()
        critic_loss = ((reward - V) ** 2).mean()
        
        # zero grading
        self.opt_actor.zero_grad()
        self.opt_critic.zero_grad()
        # back-prop-actor
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.opt_actor.step()
        self.decay_actor.step()
        # back-prop-critic
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.opt_critic.step()
        self.decay_critic.step()
        
        if not self.config.active_search:
            self.take_summary(reward.mean().tolist(), [V.mean().tolist()])
        
        # printing progress
        if self.config.print_progress:
            print('\repoch: %06d| actor_loss: %.4f,  critic_loss: %.4f,  avg_distance: %.3f,  avg_critic_out: %.3f       '
            %(self.it, actor_loss.tolist(), critic_loss.tolist(), reward.mean().tolist(), V.mean().tolist()), end='')
        
    def get_reward(self, x, indices):
        # calculate tsp distances
        distances = torch.tensor([torch.sqrt(torch.square(x[i, indices[i, 1:], :] - x[i, indices[i, :-1], :]).sum(axis=1)).sum() 
                                  for i in range(self.config.batch_size)], dtype=self.dtype, device=self.device)
        return distances
    
    def take_summary(self, reward, others=[]):
        # stacking reward and other values for learning plot
        self.reward_stack.append(reward)
        self.others_stack.append(others)
        # checking whether to store the model or not
        self.it += 1
        if self.it % self.config.save_after == 0 or self.it == self.config.epoch:
            last_reward = np.mean(self.reward_stack[-self.config.save_after:])
            if (self.ct_reward < last_reward and self.config.maximize) or (self.ct_reward > last_reward and not self.config.maximize):
                self.save_weights()
            if (self.ct_reward * self.reload_factor > last_reward and self.config.maximize) or (self.ct_reward * self.reload_factor < last_reward and not self.config.maximize):
                self.load_weights()
                
    def save_weights(self,):
        torch.save(self.actor.state_dict(), self.config.savepath + '/actor_model_weights.pt')
        torch.save(self.critic.state_dict(), self.config.savepath + '/critic_model_weights.pt')
        
    def load_weights(self,):
        self.actor.load_state_dict(torch.load(self.config.loadpath + '/actor_model_weights.pt'))
        self.critic.load_state_dict(torch.load(self.config.loadpath + '/critic_model_weights.pt'))
        
    def plot_learning(self,):
        rwd_name = 'distance'
        other_names = ['critic_out']
        same_srs_names = []  # from others stack
        
        df = pd.DataFrame(self.others_stack, columns=other_names)
        df[rwd_name] = self.reward_stack
        df[rwd_name+'_rolling'] = df[rwd_name].rolling(10).mean()
        
        same_srs_cols = [rwd_name, rwd_name+'_rolling'] + same_srs_names
        srs_cols = [c for c in df.columns if  c not in same_srs_cols]
        same_srs = [df[c] for c in same_srs_cols]
        srs = [df[c] for c in srs_cols]
        segs = 1
        msd.plot_time_series(same_srs=same_srs, srs=srs, segs=segs, fig_y=5)