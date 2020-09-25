import retro

#SHA-1: ea9bcae617fdf159b045185467ae58b2e4a48b9a

def main():
    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    # print(retro.data.list_games())
    main()


