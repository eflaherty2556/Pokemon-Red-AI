import retro
import gym
import os
import time

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
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
        env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL)
        print(env)

        print(env.action_space)
        time.sleep(3)

        env = make_vec_env(lambda: env, n_envs=1)
        # check_env(env, warn=True)
        time.sleep(3)

        model = DQN(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=25000)

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

        env.close()

    
if __name__ == "__main__":
        main()