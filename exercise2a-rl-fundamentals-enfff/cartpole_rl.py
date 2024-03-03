"""
    Robot Learning
    Exercise 2

    Reinforcement Learning 

    Polito A-Y 2023-2024
"""
import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
from agent import Agent, Policy
from utils import get_space_dim

# Setup for my Linux distro
plt.rcParams.update({
    "savefig.format": 'png',
})

import sys

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--train_episodes", "-te",type=int, default=500, help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true', help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--central_point", "-p",type=float, default=0.0, help="Point x0 to fluctuate around")
    parser.add_argument("--random_policy", action='store_true', help="Applying a random policy training")
    parser.add_argument("--reward_type", type=int, default=2, help="Use a different reward function")
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False,
          silent=False, train_run_id=0, x0=0, random_policy=False):
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    if random_policy:
        print("Training with a random policy")

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state (it's a random initial state with small values)
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            if random_policy:
                # Task 1.1
                """
                    Sample a random action from the action space
                    The allowed actions values in {0, 1}, as mentioned in https://www.gymlibrary.dev/environments/classic_control/cart_pole/
                """
                # action = random.randint(0, 1) # not the state of the art, but gets the job done.
                action = env.action_space.sample() # this is the correct way to do it


            previous_observation = observation

            # Perform the action on the environment, get new state and reward
            # note that after env._max_episode_steps the episode is over, if we stay alive that long
            observation, reward, done, info = env.step(action)

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            reward = new_reward(observation, x0)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1
        
            if done:
                print(f"last position: {observation[0]}")

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes=100, render=False):
    x0 = args.central_point
    test_reward, test_len = 0, 0

    print('Num testing episodes:', episodes)

    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
        # Task 1.2
            """
            Test on 500 timesteps
            """

            action, _ = agent.get_action(observation, evaluation=True)  # Similar to the training loop above -
                                                                        # get the action, act on the environment, save total reward
                                                                        # (evaluation=True makes the agent always return what it thinks to be
                                                                        # the best action - there is no exploration at this point)
            observation, reward, done, info = env.step(action)
            # print(observation[1])

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            reward = new_reward(observation, x0)


            if render:
                env.render()
            test_reward += reward
            test_len += 1

            if done:
                print(f"last position: {observation[0]}")

    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)

def new_reward(state, x0):
    x0 = args.central_point
    # Task 3.1
    """
        Use a different reward, overwriting the original one
        state:  the vector of the state
        x0:     the central point to fluctuate around
        This function also uses the reward_type argument
    """

    # Task 3.1
    if args.reward_type == 1:
        reward = np.exp( -5 * ( np.abs(state[0]) + np.abs(state[2]) ))

    elif args.reward_type == 2:
        # They are the same because type=1 is generalized within type=2
        # Giving equal importance to position (as close to x0) and angle (as close to 0)

        # state[0] is the position of the cart
        # state[1] is the velocity of the cart
        # state[2] is the angle of the pole
        # state[3] is the angular velocity of the pole

        reward = 1*np.exp( -3*(np.abs(state[0]-x0))) + 0.75*np.exp(-1*np.abs(state[2]))

        if (x0 > 0 and state[0] > 0) or (x0 < 0 and state[0] < 0):
            # print(f'right side!, x0: {x0}, central_point: {args.central_point}: x0, state[0]: {state[0]}')
            reward = reward + 0.3
        else:
            reward = reward - 0.3

    elif args.reward_type == 3:
        # Prova 2
        reward = 1.5*np.exp(-1*abs(1/state[1])) + np.exp(-1*abs(state[0])) + 0.01
        if np.abs(state[0]) > 2.38:
        #     # this is for the cart to avoid going out of the screen
            reward = -100 # Am I been too harsh?

        # Prova 3
        # if np.abs(state[0]) > 2.38:
        # #     # this is for the cart to avoid going out of the screen
        #     reward = -100 # Am I been too harsh?

        # if np.abs(state[1]) < 0.2 and np.abs(state[0]) < 0.2:
        #     # If slow and near the center give negative reward
        #     reward = 0.01
        # else: 
        #     reward = 1.5*np.exp(-1*abs(1/state[1])) + np.exp(-1*abs(state[0])) + 0.01
    else:
        reward = 1 # default reward

    return reward

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Task 1.2
    """
        For CartPole-v0 - change the maximum episode length
    """
    env._max_episode_steps = 500

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # User options
    print(args)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        # Train
        training_history = train(agent, env, args.train_episodes, False, args.render_training, x0=args.central_point, random_policy=args.random_policy)
        # args.train_episodes is an int, set to 500

        # Save the model
        model_file = "%s_params_t33.ai" % args.env
        torch.save(policy.state_dict(), f"/home/enf/Documents/RL/exercise2a-rl-fundamentals-enfff/prova3/{model_file}")
        torch.save(policy.state_dict(), f"/home/enf/Documents/RL/exercise2a-rl-fundamentals-enfff/{model_file}")
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
        sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
        plt.legend()
        plt.title("Reward history (%s)" % args.env)
        plt.savefig('/home/enf/Documents/RL/exercise2a-rl-fundamentals-enfff/prova3/task33.png')
        plt.show()
        print("Training finished.")
    else:
        # Test
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.train_episodes, args.render_test)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()

    if args.reward_type not in [1, 2, 3]:
        print("This kind of reward_type should only be for Task 1 and Task 2. If that's not the case, cancel the execution and use the option --reward_type=1|2|3")
        
    main(args)

