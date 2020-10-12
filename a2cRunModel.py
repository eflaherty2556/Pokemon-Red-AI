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
        print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL) #, use_restricted_actions=retro.Actions.DISCRETE
        print(env)
        
        # print(env.action_space)

        model = A2C.load("a2c_model_pkmn")
        
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

        env.close()

    
if __name__ == "__main__":
        main()