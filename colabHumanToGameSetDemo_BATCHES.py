import retro
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

"""from tkinter import filedialog
import tkinter as Tk"""
import argparse

import os
import sys

from skipWrapper import SkipLimit
from Discretizer import Discretizer

from ResizeableMatrix import ResizeableMatrix
from ResizeableArray import ResizeableArray


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
"""root = Tk.Tk()
root.withdraw()



def open_file_dialog():
    return filedialog.askopenfilename(title = "Select file",filetypes = (("movie files","*.bk2"),("all files","*.*")))

def save_file_dialog():
    return filedialog.asksaveasfilename(title = "Save as",filetypes = (("movie files","*.npz"),("all files","*.*")))"""


def main():
    movie_path = "humanDemo.bk2"
    movie = retro.Movie(movie_path)
    movie.step()

    """parser = argparse.ArgumentParser()
    parser.add_argument('--game', help='retro game to use')
    parser.add_argument('--state', help='retro state to start from')
    parser.add_argument('--scenario', help='scenario to use', default='scenario')
    args = parser.parse_args()

    if args.game is None:
        print('Please specify a game with --game <game>')
        print('Available games:')
        for game in sorted(retro.data.list_games()):
            print(game)
        sys.exit(1)

    if args.state is None:
        print('Please specify a state with --state <state>')
        print('Available states:')
        for state in sorted(retro.data.list_states(args.game)):
            print(state)
        sys.exit(1)"""

    retro.data.Integrations.add_custom_path(
                    os.path.join(SCRIPT_DIR, "custom_integrations")
            )
    #print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    # env = DummyVecEnv([lambda: SkipLimit(retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.DISCRETE), 5)]) #, use_restricted_actions=retro.Actions.DISCRETE]
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10)
    env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.ALL)
    env = Discretizer(env, [['B'], [None], ['SELECT'], ['START'],  ['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['A']])
    env = SkipLimit(env, time_between_steps=1)

    env.initial_state = movie.get_state()
    env.reset()

    run_and_create_demonstration(movie=movie, env=env)

def get_batch_name(folder:str, batch_number:int):
    batch_name = "gameDemo_"+str(batch_number)+".npz"

    return (folder + batch_name)


def run_and_create_demonstration(movie: retro.Movie, env : VecNormalize):
    demo_folder = "./Demo_Batches/"
    batch_number = 0
    movie_obs = ResizeableMatrix()
    movie_rewards = ResizeableArray()
    movie_actions = ResizeableMatrix()
    episode_counter = 0
    while movie.step():
        #Increment episode counter
        episode_counter += 1

        if episode_counter % 10000 == 0:
            print("Episode:",episode_counter)
        
        #Save batch
        if episode_counter % 50000 == 0:
            movie_episodes = [False*episode_counter]
            movie_episodes[0] = True
            
            print("Making movie_episodes")
            movie_episodes = np.array(movie_episodes)

            print("Making returns_episodes")
            movie_returns = np.array([movie_rewards.sum()])

            #Make file_name
            np.savez(get_batch_name(folder=demo_folder, batch_number=batch_number), actions=movie_actions.retrieve_matrix(), episode_returns=movie_returns, episode_starts=movie_episodes, obs=movie_obs.retrieve_matrix(), rewards=movie_rewards.retrieve_array())

            #Increment batch number
            batch_number += 1

            #Clear Previous inputs
            movie_obs = ResizeableMatrix()
            movie_rewards = ResizeableArray()
            movie_actions = ResizeableMatrix()

        #Get keys
        keys = []
        for p in range(movie.players):
            
            for i in range(env.num_buttons):
                tempKey = movie.get_key(i,0)
                if tempKey:
                    keys.append(i)

        if not keys:
            keys.append(1)
        #Append keys to actions
        movie_actions.append(keys)


        #print("Keys: ", keys)
        obs, rewards, dones, info = env.step(keys[0])

        movie_obs.append(obs)
        movie_rewards.append(rewards)
    
    
    
    movie_episodes = [False*episode_counter]
    movie_episodes[0] = True
    
    #Save batch if there was no save at the end
    if episode_counter % 50000 != 0:
        movie_episodes = [False*episode_counter]
        movie_episodes[0] = True
        
        print("Making movie_episodes")
        movie_episodes = np.array(movie_episodes)

        print("Making returns_episodes")
        movie_returns = np.array([movie_rewards.sum()])

        #Make file_name
        np.savez(get_batch_name(folder=demo_folder, batch_number=batch_number), actions=movie_actions.retrieve_matrix(), episode_returns=movie_returns, episode_starts=movie_episodes, obs=movie_obs.retrieve_matrix(), rewards=movie_rewards.retrieve_array())

        #Increment batch number
        batch_number += 1

main()


