import gym
import timer

# ODD NUMBERS ARE THE ONLY NUMBERS THAT WORK
# DON'T USE EVEN NUMBERS OR THE AGENT WILL STOP AFTER LIKE 3 SECONDS
class SkipLimit(gym.Wrapper):
    def __init__(self, env, time_between_steps):
        super(SkipLimit, self).__init__(env)
        #Create timer for counting number of steps between episodes
        self.timer = timer.StepTimer(time_limit=time_between_steps)

    def step(self, action):
        #If the timer is up, then perform regular movement
        if self.timer.time_is_done():
            observation, reward, done, info = self.env.step(action)
            self.timer.reset_timer()
        #Else just pause
        else:
            observation, reward, done, info = self.env.step(1) #None is index 1 for sum reason
        
        self.timer.increment_timer()

        return observation, reward, done, info
    
    def reset(self, **kwargs):
        self.timer.reset_timer()
        return super().reset(**kwargs)