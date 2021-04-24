import retro
import gym
import os
from Discretizer import Discretizer
from skipWrapper import SkipLimit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def makeRetroEnv(ram = False):
    obs = retro.Observations.IMAGE
    if ram:
        obs = retro.Observations.RAM
    retro.data.Integrations.add_custom_path(
                os.path.join(SCRIPT_DIR, "custom_integrations")
        )
    print("PokemonRed-GameBoy" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = retro.make("PokemonRed-GameBoy", inttype=retro.data.Integrations.ALL, obs_type=obs, use_restricted_actions=retro.Actions.ALL) #, use_restricted_actions=retro.Actions.DISCRETE
    env = Discretizer(env)
        
    env = SkipLimit(env=env, time_between_steps=5)

    return env