import gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd
import argparse

import sys

# You're welcome
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="3.1", help="Task to run") # Floats are harder to deal with
    parser.add_argument("--mode", "-m", type=str, default="TRAINING", help="Allowed values: 'TRAINING', 'TEST'")
    parser.add_argument("--eps", "-e", type=str, default="const", help="Allowed values: 'const', 'glie'")
    parser.add_argument("--episodes", type=int, default=20000, help="Positive integers only")
    parser.add_argument("--render", action='store_true', help="Render the environment during training")
    parser.add_argument("--initial_condition", type=int, default=0, help="Initial coefficient to initialize Q(s,a)")
    parser.add_argument("--path", type=str, default="q_values.npy", help="Path to the model to load or save")
    return parser.parse_args(args)

args = parse_args()
print(args)

if (args.episodes < 0):
    print("Number of episodes must be positive")
    sys.exit(1) 

if (args.task not in ["3.1", "3.2", "3.3"]):
    print('Task must be either "3.1", "3.2" or "3.3"')
    sys.exit(2)

if (args.mode not in ["TRAINING", "TEST"]):
    print('Mode must be either "TRAINING" or "TEST"')
    sys.exit(3)

# print(args.task)


np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

# Whether to perform training or use the stored .npy file
MODE = args.mode # TRAINING, TEST

test_episodes = 100
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
episodes = args.episodes
epsilon = 0 # I just need this here to make it a "global" variable
b = 2222  # choose b so that with GLIE we get an epsilon of 0.1 after 20'000 episodes

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

q_grid = args.initial_condition*np.ones((discr, discr, discr, discr, num_of_actions))

if MODE == 'TEST':
    q_grid = np.load(args.path)

def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    """Returns discrete state from continuous state"""
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, greedy=False):
    # x, v, th, av = get_cell_index(state)
    cell_index = get_cell_index(state)
    
    if greedy: # TEST -> greedy policy
        # 8c, Slides 93, find arg max wrt a of Q(s(t+1),a)
        best_action_estimated = np.argmax(q_values[cell_index])
        
        return best_action_estimated
    else: # TRAINING -> epsilon-greedy policy
        if np.random.rand() < epsilon:
            # Random action                            # Maybe done
            action_chosen = env.action_space.sample()  # TODO: choose random action with equal probability among all actions

            return action_chosen
        else:
            # Greedy action
            best_action_estimated = np.argmax(q_values[cell_index])

            return best_action_estimated


def update_q_value(old_state, action, new_state, reward, done, q_array):
    # q_array is q_grid with another name
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        # 8c, slide 95
        target_value = reward + np.max(gamma*q_array[new_cell_index])

    # 8c, slide 93
    q_old = q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action]
    q_grid[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = q_old + alpha*(target_value - q_old)

    return

    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        # 8c, slide 95
        target_value = q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] + alpha*(reward + np.max(gamma*q_array[new_cell_index[0], new_cell_index[1], new_cell_index[2], new_cell_index[3]]) - q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action])

    # 8c, slide 93
    q_grid[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = target_value

if args.task == "3.1" and args.eps == "const":
    epsilon = 0.2

if args.task == "3.3" and args.eps == "const":
    epsilon = 0

# Training loop
ep_lengths, epl_avg = [], []

# task 3.1
if args.task == "3.1" or args.task == "3.3":
    return_value, return_list = 0, [] # ret stands for return and it's the cumsum of the rewards, returns is the list of the ret

# task 3.2
# This was supposed to prove my intuition. It works but it's a mess and not worth of being shown.

# if args.task == "3.2":
#     # Printing before training
#     sim_init_q_grid = args.initial_condition*np.ones((discr, discr, discr, discr, num_of_actions))
#     sim_Vs = np.max(sim_init_q_grid, axis=4) # max_a Q(s,a)
#     # print(sim_Vs.shape) # (16, 16, 16, 16)

#     # I'm averaging over the xdot and thetadot dimensions to obtain a 2D array
#     sim_Vs = np.mean(sim_Vs, axis=(1,3))
#     # print(Vs.shape) # (16, 16)

#     plt.figure()
#     fig, axis = plt.subplots()
#     im = plt.imshow(sim_Vs)
#     plt.colorbar(im) 
#     plt.xlabel('Position')
#     plt.ylabel('Angle')
#     plt.title('(Before Training) Approximation of the State Value Function V(s)')


# if args.task == "3.2":
#     tmp,ctr = 0 , 0

for ep in range(episodes+test_episodes):
    test = ep > episodes

    # ctr += 1

    if MODE == 'TEST':
        test = True

    state, done, steps = env.reset(), False, 0

    if args.task == "3.1" and args.eps == "glie":
        epsilon = b/(b+ep)
    elif args.task == "3.3":
        epsilon = 0

    if args.task == "3.1" or args.task == "3.3":
        return_value = 0 # reset return value
    
    while not done:
        action = get_action(state, q_grid, greedy=test)
        new_state, reward, done, _ = env.step(action)
        if test:
            # Test
            if args.render:
                env.render()
        else:
            # Training
            update_q_value(state, action, new_state, reward, done, q_grid)
            
        state = new_state
        steps += 1

        if args.task == "3.1" or args.task == "3.3":
            return_value += reward

    if args.task == "3.1" or args.task == "3.3":
        return_list.append(return_value)

    ep_lengths.append(steps)

    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, Epsilon {}: {}, average timesteps: {:.2f}".format(ep, args.eps, epsilon, np.mean(ep_lengths[max(0, ep-200):])))

    # if args.task == "3.2" and ep % 100 == 0 and tmp == 0 and ctr == 2:
    #     tmp = 1
    #     sim_init_q_grid = args.initial_condition*np.ones((discr, discr, discr, discr, num_of_actions))
    #     sim_Vs = np.max(sim_init_q_grid, axis=4) # max_a Q(s,a)
    #     # print(sim_Vs.shape) # (16, 16, 16, 16)

    #     # I'm averaging over the xdot and thetadot dimensions to obtain a 2D array
    #     sim_Vs = np.mean(sim_Vs, axis=(1,3))
    #     # print(Vs.shape) # (16, 16)

    #     plt.figure(2)
    #     fig, axis = plt.subplots()
    #     im = plt.imshow(sim_Vs)
    #     plt.colorbar(im) 
    #     plt.xlabel('Position')
    #     plt.ylabel('Angle')
    #     plt.title('(During Training) Approximation of the State Value Function V(s)')
        

if args.task == "3.1" or args.task == "3.3":
    fig, ax = plt.subplots()
    ax.plot(range(episodes+test_episodes), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.grid(True)
    if MODE == "TRAINING":
        plt.title(f"Training rewards for epsilon {args.eps}")
    elif MODE == "TEST":
        plt.title(f"Test rewards for epsilon {args.eps} ({epsilon})")
    else:
        plt.title("Something went wrong check the code")
    plt.show()
elif args.task == "3.2":
    # I hope what I'm doing is right
    # State value function approximation V(s)

    Vs = np.max(q_grid, axis=4) # max_a Q(s,a)
    print(Vs.shape) # (16, 16, 16, 16)

    # I'm averaging over the xdot and thetadot dimensions to obtain a 2D array
    Vs = np.mean(Vs, axis=(1,3))
    # print(Vs.shape) # (16, 16)

    plt.figure()
    fig, axis = plt.subplots()
    im = plt.imshow(Vs)
    plt.colorbar(im) 
    plt.xlabel('Position')
    plt.ylabel('Angle')
    plt.title('Approximation of the State Value Function V(s)')
    plt.show()
if MODE == 'TEST':
    sys.exit()

# Save the Q-value array
np.save(args.path, q_grid)