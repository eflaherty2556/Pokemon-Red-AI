import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import retro
import gym
import time

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env

from skipWrapper import SkipLimit
from Discretizer import Discretizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join(SCRIPT_DIR, "../")

def main():
    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))

    print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.ALL)
    env = Discretizer(env)
    
    #Wrap enviornment for skip limit
    env = SkipLimit(env=env, time_between_steps=3)    

    model = DQN(MlpPolicy, env, verbose=1)

    print("STARTING Training!")
    start_time = time.time()

    model.learn(total_timesteps=5000)
    print("TRAINING COMPLETE! Time elapsed: ", str(time.time()-start_time))

   
    
if __name__ == "__main__":
        main()