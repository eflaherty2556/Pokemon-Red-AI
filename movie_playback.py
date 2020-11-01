import retro
import gym
import os
import time

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env

# party size = d163
# money = d347

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
        retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
        #print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL) #, use_restricted_actions=retro.Actions.DISCRETE
        #print(env)
        
        # print(env.action_space)


        

        movie = retro.Movie('PokemonRed-GameBoy-red-start-000000.bk2')
        env.initial_state = movie.get_state()
        
        env.reset()
        while movie.step():
            keys = []
            for p in range(movie.players):
                for i in range(env.num_buttons):
                    keys.append(movie.get_key(i,p))
            env.step(keys)
                

        env.close()

    
if __name__ == "__main__":
        main()