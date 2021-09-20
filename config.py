import argparse
import os


def prepare_tsp_config(show=True, jupyter=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for modeling')
    parser.add_argument('--problem_size', type=int, default=20, help='problem size')
    parser.add_argument('--dimension', type=int, default=2, help='problem dimension')
    parser.add_argument('--lr', type=float, default=.0001, help='learning rate')
    parser.add_argument('--num_test', type=int, default=100, help='number of test cases')
    parser.add_argument('--start_index', type=int, default=0, help='starting point')
    parser.add_argument('--device', type=str, default='cuda', help='computation device gpu/cpu')
    parser.add_argument('--hidden1', type=str, default=256, help='hidden layer 1 units (embedding hidden size)')
    parser.add_argument('--hidden2', type=str, default=128, help='hidden layer 2 units')
    parser.add_argument('--c', type=float, default=10, help='C value to control range of tanh output during attending')
    parser.add_argument('--T', type=float, default=3, help='temperature value T according to the paper from Bello')
    parser.add_argument('--n_glimpse', type=int, default=1, help='number of glimpse computation')
    parser.add_argument('--print_progress', type=bool, default=True, help='printing progress during training')
    parser.add_argument('--active_search', type=bool, default=False, help='whether to run active search or not')
    parser.add_argument('--savepath', type=str, default='trained_model', help='directory path to store the trained models')
    parser.add_argument('--loadpath', type=str, default='trained_model', help='directory path to load the trained models')
    parser.add_argument('--save_after', type=int, default=100, 
                        help='after how many epochs the model will check to store models during training')
    parser.add_argument('--resume_training', action='store_true',
                        default=False, help='Whether to resume training from a previous training or not')
    parser.add_argument('--iteration', type=int, default=1, help='number of iterations')
    parser.add_argument('--maximize', action='store_true', default=False, help='whether it is a maximization problem or not')
    parser.add_argument('--greedy', action='store_true', default=False, help='whether to use greedy action or sampling')
    parser.add_argument('--epoch', type=int, default=10000, help='number of epochs')
    parser.add_argument('--inference', action='store_true', default=False, help='inference mode or not')
    parser.add_argument('--lr_step', type=int, default=50, help='number of steps to reduce learning rate')
    parser.add_argument('--lr_step_decay', type=float, default=.999, help='factor to reduce learning rate')
    parser.add_argument('--alpha', type=float, default=.99, help='update factor for average baseline')
    parser.add_argument('--num_it', type=int, default=1, help='number of iteration in inference time')
    parser.add_argument('--show_sample', action='store_true', default=False, help='whether to show sample output or not')
    
    
    if jupyter:
        config = parser.parse_args([])
    else:
        config = parser.parse_args()
    config.savepath += '/size_%d'%config.problem_size
    config.loadpath += '/size_%d'%config.problem_size
    if not os.path.exists(config.savepath): os.makedirs(config.savepath)
    if not os.path.exists(config.loadpath): os.makedirs(config.loadpath)
    if show: print(config.__dict__)
    return config