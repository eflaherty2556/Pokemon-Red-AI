import retro
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
from tkinter import filedialog
import tkinter as Tk
import os
from skipWrapper import SkipLimit
root = Tk.Tk()
root.withdraw()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def open_file_dialog():
    return filedialog.askopenfilename(title = "Select file",filetypes = (("movie files","*.bk2"),("all files","*.*")))

def save_file_dialog():
    return filedialog.asksaveasfilename(title = "Select file",filetypes = (("movie files","*.bk2"),("all files","*.*")))

def main():
    movie_path = open_file_dialog()
    movie = retro.Movie(movie_path)
    movie.step()

    retro.data.Integrations.add_custom_path(
                    os.path.join(SCRIPT_DIR, "custom_integrations")
            )
    print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    # env = DummyVecEnv([lambda: SkipLimit(retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.DISCRETE), 5)]) #, use_restricted_actions=retro.Actions.DISCRETE]
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10)
    env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.DISCRETE)

    env.initial_state = movie.get_state()
    env.reset()

    run_and_create_demonstration(movie=movie, env=env)


def run_and_create_demonstration(movie: retro.Movie, env : VecNormalize):

    movie_obs = []
    movie_rewards = []
    movie_actions = []
    episode_counter = 0
    while movie.step():
        #Increment episode counter
        episode_counter += 1

        #Get keys
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                tempKey = movie.get_key(i,p)
                if tempKey:
                    keys.append([env.buttons[i]])

        #Append keys to actions
        movie_actions.append(keys)

        print("Keys: ", keys)
        obs, rewards, dones, info = env.step(keys)

        movie_obs.append(obs)
        movie_rewards.append(rewards)
    
    movie_episodes = [False*episode_counter]
    movie_episodes[0] = True
    
    movie_episodes = np.array(movie_episodes)
    movie_returns = np.array(list(map(sum, movie_rewards)))
    movie_rewards = np.array(list(map(np.array, movie_rewards)))
    movie_actions = np.array(list(map(np.array, movie_actions)))
    movie_obs = np.array(list(map(np.array, movie_obs)))

    np.savez(save_file_dialog(), actions=movie_actions, episode_returns=movie_returns, episode_starts=movie_episodes, obs=movie_obs, rewards=movie_rewards)
        

main()


