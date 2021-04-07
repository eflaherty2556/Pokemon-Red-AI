import retro
import gym
import os
import time
import retro.enums

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecNormalize
from skipWrapper import SkipLimit

from Discretizer import Discretizer

#
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="cnn")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    model_name = "a2c_mlp_ram_testEval" #for saving and logging with tensorboard

    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))

    print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.RAM, use_restricted_actions=retro.Actions.ALL, record=True)
    env = Discretizer(env)

    print("\n\nACTION Space:\n\n", env.action_space)
    
    #Wrap enviornment for skip limit
    env = SkipLimit(env=env, time_between_steps=3)
    #obs_type=retro.Observations.RAM #see https://retro.readthedocs.io/en/latest/python.html#observations
    
    

    vec_env = make_vec_env(lambda: env, n_envs=32)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    # time.sleep(3)    

    model = A2C(MlpPolicy, vec_env, verbose=1, tensorboard_log="./pokemon-red-tensorboard/", learning_rate=0.001)

    # pretrain? https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html

    start_time = time.time()
    model.learn(total_timesteps=50000, tb_log_name=model_name)
    print("TRAINING COMPLETE! Time elapsed: ", str(time.time()-start_time))

    print("Saving model...")
    model.save(model_name)


    
    start_time = time.time()
    printed_done = False
    # sampled_info = False

    # print("Evaluating now...")
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, render=False, return_episode_rewards=True)
    # print("done evaluating! mean reward: ", mean_reward)
    # print("done evaluating! std reward: ", std_reward)

    curent_step = 0
    movie_step_length = 10000

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        # print("ACTION: ", action)
        obs, rewards, dones, info = env.step(action)
        env.render()
        # if not sampled_info:
        #     print("Info:\n", info, "\n</info>")
        #     sampled_info = True

        if dones and not printed_done:
            print("Success! time elapsed: ", str(time.time()-start_time))
            printed_done = True

        if curent_step < movie_step_length:
            curent_step += 1
        else:
            break

    env.close()

    
if __name__ == "__main__":
        main()

