import retro
import os

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

        printCounter = 0
        obs = env.reset()
        while True:
            obs, rew, done, info = env.step(env.action_space.sample())
            env.render()

            if printCounter % 10000:
                print("reward: ", rew)

            if done:
                obs = env.reset()

            printCounter += 1

        env.close()

    
if __name__ == "__main__":
        main()