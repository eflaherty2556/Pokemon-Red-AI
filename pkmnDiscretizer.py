class pokemonRedDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Space Invaders game.
    """
    def __init__(self, env):
        super(pokemonRedDiscretizer, self).__init__(env)
        buttons = ["A", "LEFT", "RIGHT"]
        actions = [["BUTTON"], ['LEFT'], ['RIGHT'], ["BUTTON","LEFT"], ["BUTTON", "RIGHT"]]
        self._actions = []
        """
        What we do in this loop:
        For each action in actions
            - Create an array of 3 False (3 = nb of buttons)
            For each button in action: (for instance ['LEFT']) we need to make that left button index = True
                - Then the button index = LEFT = True
            In fact at the end we will have an array where each array is an action and each elements True of this array
            are the buttons clicked.
        """
        for action in actions:
            arr = np.array([False] * 3)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()