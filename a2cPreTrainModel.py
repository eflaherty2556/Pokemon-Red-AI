import retro
import gym
import os
import time
import sys

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines import A2C
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecNormalize
from stable_baselines.gail import ExpertDataset
from skipWrapper import SkipLimit
from Discretizer import Discretizer

from makeRetroEnv import makeRetroEnv

import argparse


#Not Used
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="cnn")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def train_model(n_vec = 4, time_steps = 4000, epochs = 500):
        """ retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
        print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
        env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=retro.Observations.IMAGE, use_restricted_actions=retro.Actions.ALL) #, use_restricted_actions=retro.Actions.DISCRETE
        env = Discretizer(env)
        print(env)
        
        env = SkipLimit(env=env, time_between_steps=5) """
        env = makeRetroEnv(ram=False)
        vec_env = make_vec_env(lambda: env, n_envs=n_vec)
        #vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10)
         

        expert_dataset = ExpertDataset(expert_path='./gameDemo.npz')
        model = A2C(MlpPolicy, vec_env, verbose=1, tensorboard_log="./pokemon-red-tensorboard/", lr_schedule='middle_drop', vf_coef=0.3, max_grad_norm=0.55, learning_rate=7e-5)

        start_time = time.time()

        model.pretrain(expert_dataset, n_epochs=epochs)
        
        print("PRETRAINING COMPLETE! Time elapsed: ", str(time.time()-start_time))
        
        """ #Save env stats
        print("Saving env stats...")
        vec_env.save("a2c_env_stats_pkmn_pretrain.pk1") """

        #Save model
        print("Saving model...")
        model.save("a2c_mlp_5M_pretrain")

        #Save env stats
        #print("Saving env stats...")
        #vec_env.save("a2c_env_stats_pkmn.pk1")

def main():
     parser = argparse.ArgumentParser(description="Train pretrained model")
     parser.add_argument("-e", "--epochs", help="Number of epochs")
     parser.add_argument('--n_vec', help="Number of vectors in enviornment", default=32)
     args = parser.parse_args()
     train_model(n_vec=int(args.n_vec), epochs=int(args.epochs))
if __name__ == "__main__":
        main()