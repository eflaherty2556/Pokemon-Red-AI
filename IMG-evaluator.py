import sys
import matplotlib.pyplot as plot
import retro
import gym
import os
import time
import retro.enums

from datetime import datetime
from datetime import date
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    model_name = sys.argv[1]
    eval_time = int(sys.argv[2])

    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))

    print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE, use_restricted_actions=retro.Actions.ALL)
    env = Discretizer(env)
    
    #Wrap enviornment for skip limit
    env = SkipLimit(env=env, time_between_steps=3)    

    vec_env = make_vec_env(lambda: env, n_envs=32)
    # vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    model = A2C.load(model_name)

    print("Starting evaluation now...")

    start_time = time.time()
    cumulative_reward = 0
    episode_reward = 0
    rewards = []
    cumulative_rewards = []

    obs = env.reset()
    for i in range(eval_time):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # env.render()
        cumulative_reward += reward
        episode_reward += reward
        rewards.append(reward)
        cumulative_rewards.append(cumulative_reward)

        #in case the agent completes an episode
        if done:
            print("Finished episode with reward of ", episode_reward, " at ", i, " timesteps.")
            episode_reward = 0
            obs = env.reset()

    env.close()
    time_elapsed = time.time()-start_time

    print("--------------------SUMMARY--------------------")
    print("-Time elapsed: ", time_elapsed)
    print("-Number of evaluation steps: ", eval_time)
    print("-Model: ", model_name)
    print("--------------------RESULTS--------------------")
    print("-Cumulative reward: ", cumulative_reward)
    print("-Reward per step: ", cumulative_reward/eval_time)
    print("--------------------GRAPHS---------------------")
    # plot.figure(figsize=(9, 12))
    plot.figure()
    plot.style.use('ggplot')

    # plot.subplot(311)
    plot.plot(rewards)
    plot.title("Rewards")
    plot.xlabel("Timestep")
    plot.ylabel("Reward")
    plot.grid(True)
    save(model_name + "-eval-" +"rewards_")
    plot.show()

    # plot.subplot(313)
    plot.plot(cumulative_rewards)
    plot.title("Cumulative Rewards")
    plot.xlabel("Timestep")
    plot.ylabel("Reward")
    plot.grid(True)
    save(model_name + "-eval-" +"cumulative_")
    plot.show()


def save(prefix):
    output_dir = SCRIPT_DIR+"/eval_figures"
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    now = datetime.now()
    today = date.today()

    current_time = now.strftime("%H-%M-%S")
    current_day = today.strftime("%m-%d-%y--")


    figure_name = "eval_figures/" + prefix + current_day + current_time + ".png"
    plot.savefig(fname=figure_name)
    print("Plot output to: ", str(SCRIPT_DIR+"/" + figure_name))
    
if __name__ == "__main__":
        main()

