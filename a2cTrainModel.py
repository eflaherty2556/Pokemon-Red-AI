import retro
import gym
import os
import time
import retro.enums
import sys

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy, register_policy


# party size = d163
# money = d347

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

def train_model(n_vec = 4, time_steps = 2000):
        retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
        print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL) #, use_restricted_actions=retro.Actions.DISCRETE
        print(env)
        
        # print(env.action_space)

        vec_env = make_vec_env(lambda: env, n_envs=n_vec)
        # time.sleep(3)    

        model = A2C(CustomPolicy, vec_env, verbose=1)

        start_time = time.time()
        model.learn(total_timesteps=time_steps)
        print("TRAINING COMPLETE! Time elapsed: ", str(time.time()-start_time))
        
        model.save("a2c_model_pkmn")

def main():
        if len(sys.argv) - 1 == 0:
                train_model(n_vec=4, time_steps=100000)   
        else:
                train_model(n_vec=4, time_steps=int(sys.argv[1]))

    
if __name__ == "__main__":
        main()