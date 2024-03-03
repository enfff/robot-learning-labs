"""
    Robot Learning
    Exercise 2

    Linear Quadratic Regulator

    Polito A-Y 2023-2024
"""
import gym
import numpy as np
from scipy import linalg     # get riccati solver
import argparse
import matplotlib.pyplot as plt
import sys
from utils import get_space_dim, set_seed
import pdb 
import time

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--time_sleep", action='store_true',
                        help="Add timer for visualizing rendering with a slower frame rate")
    parser.add_argument("--mode", type=str, default="control",
                        help="Type of test ['control', 'multiple_R']")
    return parser.parse_args(args)

def linerized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    a = g/(lp*(4.0/3 - mp/(mp+mk)))
    # state matrix
    A = np.array([[0, 1, 0, 0],
                [0, 0, a, 0],
                [0, 0, 0, 1],
                [0, 0, a, 0]])

    # input matrix
    b = -1/(lp*(4.0/3 - mp/(mp+mk)))
    B = np.array([[0], [1/mt], [0], [b]])
    return A, B

def optimal_controller(A, B, R_value=1):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=False):
    """
    Vary the value of R within the range [0.01, 0.1, 10, 100] and plot the forces 
    """

    obs = env.reset()    # Reset the environment for a new episode
    
    A, B = linerized_cartpole_system(mp, mk, l, g)
    ctr = 0 # not proud of this but this ain't a programming class!!

    fig, ax = plt.subplots(4)

    for R_val in [0.01, 0.1, 10, 100]:
        terminate = False
        Done = False

        K = optimal_controller(A, B, R_val)    # Compute the optimal controller for the current R value
        forces = []
        print('res forces')

        for i in range(1000):
            env.render()
            if time_sleep:
                time.sleep(.1)
            
            # get force direction (action) and force value (force)
            action, force = apply_state_controller(K, obs)

            forces.append(force) # storing forces
            
            # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
            abs_force = abs(float(np.clip(force, -10, 10)))
            # this gives a warning but its not up to me to fix it
            
            # change magnitute of the applied force in CartPole
            env.env.force_mag = abs_force

            # apply action
            obs, reward, done, _ = env.step(action)

            if i >= 399:
                terminate = True
            
            if terminate and done:
                print(f'Terminated after {i+1} iterations.')
                break

        print('i reached here')

        if forces[0] > 0:
            ax[ctr].plot(range(0, len(forces)), [ -f for f in forces]) # inverts the sign of each value in the list
        else:
            ax[ctr].plot(range(0, len(forces)), forces)

        # ax[ctr].plot(range(0, len(forces)), forces)
        print(f'plottetd ax{ctr}')
        ctr += 1

    plt.show()
    return

def control(env, mp, mk, l, g, time_sleep=False, terminate=False):
    """
    Control using LQR
    """

    obs = env.reset()    # Reset the environment for a new episode
    
    A, B = linerized_cartpole_system(mp, mk, l, g)
    K = optimal_controller(A, B)    # Re-compute the optimal controller for the current R value

    x = []
    xdot = []
    theta = []
    thetadot = []

    for i in range(1000):

        env.render()
        if time_sleep:
            time.sleep(.1)
        
        # get force direction (action) and force value (force)
        action, force = apply_state_controller(K, obs)
        
        # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
        abs_force = abs(float(np.clip(force, -10, 10)))
        
        # change magnitute of the applied force in CartPole
        env.env.force_mag = abs_force

        # apply action
        obs, reward, done, _ = env.step(action)

        x.append(obs[0])
        xdot.append(obs[1])
        theta.append(obs[2])
        thetadot.append(obs[3])

        if i >= 399:
            terminate = True
        
        if terminate and done:
            print(f'Terminated after {i+1} iterations.')
            break

    fig, ax = plt.subplots(4)

    ax[0].plot(range(0, len(x)), x, label=["", "x"])
    ax[1].plot(range(0, len(xdot)), xdot, label=["", "xdot"])
    ax[2].plot(range(0, len(theta)), theta, label=["", "theta"])
    ax[3].plot(range(0, len(thetadot)), thetadot, label=["", "thetadot"])

    plt.show()
    # Confrontando ad occhio, convergono a zero dopo la 34esima iterazione.
    # X è sempre nel range
    # X dot converge dopo 34 iterazioni
    # Theta è sempre nel range
    # Theta dot converge dopo 7 iterazioni
    # Vedi images/task1.png

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print("Environment:", args.env)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    set_seed(args.seed)    # seed for reproducibility
    env.env.seed(args.seed)
    
    mp, mk, l, g = env.masspole, env.masscart, env.length, env.gravity

    if args.mode == "control":
        control(env, mp, mk, l, g, args.time_sleep)
    elif args.mode == "multiple_R":
        multiple_R(env, mp, mk, l, g, args.time_sleep)

    env.close()

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

