from makeRetroEnv import makeRetroEnv
import retro
import gym
import os
import time
import argparse

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from skipWrapper import SkipLimit
from Discretizer import Discretizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description="Train pretrained model")
    parser.add_argument("-m", "--model", help="Model directory")
    args = parser.parse_args()

    retro.data.Integrations.add_custom_path(
            os.path.join(SCRIPT_DIR, "custom_integrations")
    )
    print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    # env = DummyVecEnv([lambda: SkipLimit(retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.ALL), 3)]) #, use_restricted_actions=retro.Actions.DISCRETE]
    """ env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.ALL) #, use_restricted_actions=retro.Actions.DISCRETE
    env = Discretizer(env)
    env = SkipLimit(env=env, time_between_steps=5) """
    env = makeRetroEnv()
    vec_env = DummyVecEnv([lambda: env])
    #vec_env = VecNormalize.load("a2c_env_stats_pkmn.pk1", vec_env)

    done_printed = False
    time_start = time.time()

    #Load model
    model = A2C.load(args.model)
    
    cumulative_reward = 0
    timesteps = 0

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, dones, info = vec_env.step(action)
        vec_env.render()

        cumulative_reward += reward
        timesteps += 1

        if timesteps % 2500 == 0:
            print("\n-=Timestep ", timesteps, "=-")
            print("Cumulative Reward: ", cumulative_reward)
            print("Reward Per Step:", (cumulative_reward/timesteps))

    vec_env.close()

    
if __name__ == "__main__":
        main()