import retro
import gym
import os
import time
import retro.enums

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy, register_policy

#
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
        retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
        print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.DISCRETE) #, use_restricted_actions=retro.Actions.DISCRETE
        print(env)
        
        # print(env.action_space)

        vec_env = make_vec_env(lambda: env, n_envs=4)
        # time.sleep(3)    

        model = A2C(CustomPolicy, vec_env, verbose=1)

        start_time = time.time()
        model.learn(total_timesteps=50000)
        print("TRAINING COMPLETE! Time elapsed: ", str(time.time()-start_time))
        
        print("Attempting to get first pokemon!")
        start_time = time.time()
        printed_done = False
        # sampled_info = False

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            # if not sampled_info:
            #     print("Info:\n", info, "\n</info>")
            #     sampled_info = True

            if dones and not printed_done:
                print("Success! time elapsed: ", str(time.time()-start_time))
                printed_done = True

        env.close()

    
if __name__ == "__main__":
        main()