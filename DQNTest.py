import retro
import gym
import os
import time

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines import DQN
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
        env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, use_restricted_actions=retro.Actions.DISCRETE)
        print(env)

        # print(env.action_space)
        # time.sleep(3)

        # env = make_vec_env(lambda: env, n_envs=1)
        # check_env(env, warn=True)
        # time.sleep(3)

        model = DQN(CnnLnLstmPolicy, env, verbose=1, tensorboard_log="./pokemon-red-tensorboard/")

        print("STARTING Training!!!")
        start_time = time.time()
        model.learn(total_timesteps=10000, tb_log_name="DQN-CNNLnLstm")
        print("TRAINING COMPLETE! Time elapsed: ", str(time.time()-start_time))

        print("Attempting to get first pokemon!")
        start_time = time.time()
        printed_done = False
        sampled_info = False

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

            if not sampled_info:
                print("Here's the info that the AI uses:\n")
                print("obs:\n", obs, "\n</obs>\n")
                print("rewards:\n", rewards, "\n</rewards>\n")
                print("dones:\n", dones, "\n</dones>\n")                
                print("Info:\n", info, "\n</info>\n")
                sampled_info = True

            if dones and not printed_done:
                print("Success! time elapsed: ", str(time.time()-start_time))
                printed_done = True


        env.close()

    
if __name__ == "__main__":
        main()