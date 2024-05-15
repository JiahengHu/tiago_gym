import gym
from gymnasium import spaces
import numpy as np
import time

class TiagoPointHeadWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.head_goal = np.array([0.6, 0.6])
        self.max_steps = 10
        self.cur_steps = 0

    @property
    def action_space(self):
        return spaces.Discrete(5)

    def reset(self, *args, **kwargs):
        self.cur_steps = 0
        obs, info = self.env.reset()
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step({'head': action})
        head_joints = obs['head']

        reward =  0.5 - np.linalg.norm(self.head_goal - head_joints)

        self.cur_steps += 1

        truncated = False
        if self.max_steps <= self.cur_steps:
            truncated = True

        return obs, reward, terminated, truncated, info
    
from tiago_gym.utils.display_utils import FullscreenImageDisplay
class DisplayImageWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.display = FullscreenImageDisplay(update_interval=1, monitor_id=0)

        blue_img = np.zeros((256, 256, 3), dtype=np.uint8)
        blue_img[..., 2] = 255

        green_img = np.zeros((256, 256, 3), dtype=np.uint8)
        green_img[..., 1] = 255

        img = np.concatenate((blue_img, green_img), axis=1)
        self.img1 = self.display.get_resized_image(img)

        img = np.concatenate((green_img, blue_img), axis=1)
        self.img2 = self.display.get_resized_image(img)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset()
        
        random_var = np.random.randint(0, 2)
        if random_var == 0:
            self.display.set_update(self.img1)
        else:
            self.display.set_update(self.img2)
        time.sleep(1)
        
        return obs, info