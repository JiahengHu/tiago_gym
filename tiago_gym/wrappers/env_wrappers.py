import gym
from gymnasium import spaces
import numpy as np
import time
import cv2
from collections import OrderedDict
from tiago_gym.utils.general_utils import AttrDict, copy_np_dict

def merge_dicts(d1, d2):
    for k in d1.keys():
        assert k not in d2, f'found same key ({k}) in both dicts'
        d2[k] = d1[k]
    return d2


class SimAndRealUncertaintyAwareWrapper(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        from l2l.config.env.robosuite.skill_color_pnp import env_config
        from l2l.utils.general_utils import get_env_from_config
        from l2l.wrappers.robosuite_wrappers import RobosuiteGymWrapper

        self.sim_env = get_env_from_config(AttrDict(env_config=env_config, wrappers=[RobosuiteGymWrapper]))
    
        self.encoder = None

        self.mode = 'robot'
        self.camera_step = 0
        self.max_camera_steps = 20
        self.max_camera_steps_per_stage = 20

        self.new_episode = True
        self.full_obs = None

        self.z, self.reward = [], []

    @property   
    def observation_space(self):
        ob_space = OrderedDict()
        sim_ob_space = self.sim_env.observation_space
        real_ob_space = self.env.observation_space

        ob_space['privileged_info'] = sim_ob_space['privileged_info']
        for k in real_ob_space:
            ob_space[k] = real_ob_space[k]
        return spaces.Dict(ob_space)
        
    @property
    def action_space(self):
        return spaces.Discrete(3)
    
    def set_encoder(self, encoder):
        self.encoder = encoder
    
    def update_dict(self, d):
        '''
            Overwrite keys of d1 with those of d2
        '''
        for k in d.keys():
            assert k in self.full_obs.keys(), f'Key {k} not found in dictionary d1'
            self.full_obs[k] = d[k]
    
    def hard_reset(self, **kwargs):
        obs_real, info_real = self.env.reset(**kwargs)
        obs_sim, info_sim = self.sim_env.reset(**kwargs)

        obs = merge_dicts(obs_real, obs_sim)
        info = merge_dicts(info_real, info_sim)

        return obs, info

    def reset(self, **kwargs):
        if self.new_episode:
            self.full_obs, info = self.hard_reset(**kwargs)
            self.camera_step = 0
        else:
            info = {}

        self.mode = 'robot'
        _, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(0, False, False, {})
        if terminated or truncated:
            self.full_obs, info = self.hard_reset(**kwargs)
            self.camera_step = 0
            _, reward, terminated, truncated, _ = self.rollout_robot_until_uncertain(0, False, False, {})
            
        
        self.mode = 'camera'

        if len(self.z)> 0:
            self.z = np.array(self.z)
            mx = np.max(self.z, axis=0)
            mn = np.min(self.z, axis=0)
            print(f"({mx[0]:0,.2f}, {mx[1]:0,.2f}), ({mn[0]:0,.2f}, {mn[1]:0,.2f}), {np.mean(self.reward):0,.2f}")
            
        self.z, self.reward = [], []
        
        return copy_np_dict(self.full_obs), info

    def rollout_robot_until_uncertain(self, reward, terminated, truncated, info):
        while self.mode == 'robot':
            if self.encoder is None:
                output = (len(self.sim_env.unwrapped.skills)-1, False)
                loss = 1
            else:
                output = self.encoder.get_action_and_uncertainty(self.full_obs)
                loss = self.encoder.get_reward(self.full_obs)
            robot_action, uncertain = output[0], output[1]
            
            for _ in range(5):
                cv2.imshow('obs', np.concatenate((self.full_obs['agentview_image'],self.full_obs['tiago_head_image']), axis=1)/255)
                cv2.waitKey(1)
            
            if loss > 0.5:
                self.mode = 'camera'
                break
            
            sim_obs, reward, terminated, truncated, info = self.sim_env.step(robot_action)
            self.update_dict(sim_obs)

            if terminated or truncated:
                break

        return self.full_obs, reward, terminated, truncated, info

    def step(self, action):
        assert self.mode == 'camera', 'set to camera mode before stepping'
        
        real_obs, _, _, _, info = self.env.step({'head': action})
        self.update_dict(real_obs)
        truncated = False
        terminated = False
        loss = self.encoder.get_reward(copy_np_dict(self.full_obs))
        print(action, f'{loss:0,.2f}')
        reward = np.clip(1-loss, -1, 1)

        self.z.append(self.full_obs['head'])
        self.reward.append(reward)

        self.camera_step += 1
        if self.camera_step % self.max_camera_steps_per_stage == 0:
            terminated = True
            self.new_episode = False

        if self.camera_step >= self.max_camera_steps:
            terminated = True
            truncated = False
            self.new_episode = True

        cv2.imshow('obs', np.concatenate((self.full_obs['agentview_image'],self.full_obs['tiago_head_image']), axis=1)/255)
        cv2.waitKey(1)

        return copy_np_dict(self.full_obs), reward, terminated, truncated, info

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
        
        if obs['privileged_info'][0]==1 :
            # print('blue on left')
            self.display.set_update(self.img1)
        else:
            # print('blue on right')
            self.display.set_update(self.img2)
        time.sleep(1)
        
        return obs, info
    
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
