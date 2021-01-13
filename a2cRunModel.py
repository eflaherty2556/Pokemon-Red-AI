import retro
import gym
import os
import time

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from skipWrapper import SkipLimit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
        retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
        print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = DummyVecEnv([lambda: SkipLimit(retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.DISCRETE), 5)]) #, use_restricted_actions=retro.Actions.DISCRETE]
        env = VecNormalize.load("a2c_env_stats_pkmn.pk1", env)
        print(env)

        done_printed = False
        time_start = time.time()

        #Load model
        model = A2C.load("a2c_mlp_5M")
        
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            if dones and not done_printed:
                    print("Time to completion:", str(time.time() - time_start))
                    done_printed = True

        env.close()

    
if __name__ == "__main__":
        main()