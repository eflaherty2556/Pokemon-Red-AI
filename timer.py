
class StepTimer():
    def __init__(self, time_limit = 15) -> None:
        self.time_limit = time_limit
        self.reset_timer()
    
    def reset_timer(self):
        self.current_time = 0
        self.timer_done = False
    
    def increment_timer(self):
        self.current_time += 1
        if self.current_time >= self.time_limit:
            self.timer_done = True
    
    def time_is_done(self):
        return self.timer_done