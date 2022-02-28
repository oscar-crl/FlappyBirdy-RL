from gym import Env
from gym.spaces import Discrete, Box

import numpy as np


class CrappyBirdEnv(Env):

    def __init__(self, root, game_window):

        self.action_space = Discrete(2)
        # 0 = Y position of the bird | 1 = Distance from the bird to the next pipe | 2 = Y position of the hole
        self.observation_space = Box(-np.inf, np.inf, shape=(2,))

        # print(self.observation_space.sample())
        self.root = root
        self.game_window = game_window

    def _get_observation(self, action):
        h_dist, v_dist = self.game_window.tick(action)
        h_dist /= (self.game_window.screen_size[0] - 250)
        v_dist /= self.game_window.screen_size[1]
        return np.array([h_dist, v_dist])

    def step(self, action):

        obs = self._get_observation(action)

        return obs, 1, self.game_window.stop, {}

    def render(self, **kwargs):
        self.root.update_idletasks()
        self.root.update()

    def reset(self):
        # 0 = Y position of the bird | 1 = Distance from the bird to the next pipe | 2 = Y position of the hole
        # self.observation_space = MultiDiscrete([641, 481, 641])
        self.game_window.reset()
        return self._get_observation(0)

    def close(self):
        exit(0)
