import os
import pickle
import torch
import numpy as np
import cv2
from torchvision import transforms

class TiagoActioUnnormalizePolicy:

    def __init__(self, policy, action_mean, action_std):
        self.policy = policy # this is a RolloutPolicy in RoboMimic
        self.action_mean = action_mean
        self.action_std = action_std

        # dims to unnormalize - won't unnormalize last len(action) - len(action_mean) dims
        self.unnormalize_dims = len(action_mean)

    def unnormalize(self, action):
        action = action * self.action_std + self.action_mean
        return action

    def get_action(self, obs):
        action = self.policy(obs)
        action[:self.unnormalize_dims] = self.unnormalize(action[:self.unnormalize_dims])

        return action
    
    def start_episode(self):
        self.policy.start_episode()

class RoboMimicPolicy:

    def __init__(self, policy, data_stats_path, unnormalize_action_dims):
        self.policy = policy # this is a RolloutPolicy in RoboMimic
        self.data_stats = self.load_data_stats(data_stats_path)

        # first dims to unnormalize
        self.unnormalize_action_dims = unnormalize_action_dims

    def unnormalize_actions(self, action):
        action = action * self.data_stats['std']['actions'][:self.unnormalize_action_dims] + self.data_stats['mean']['actions'][:self.unnormalize_action_dims]
        return action

    def normalize_obs(self, obs):
        for k in obs.keys():
            if k in self.data_stats['mean']:
                obs[k] = (obs[k] - self.data_stats['mean'][k])/self.data_stats['std'][k]
        return obs

    def get_action(self, obs):
        obs = self.normalize_obs(obs)
        action = self.policy(obs)
        action[:self.unnormalize_action_dims] = self.unnormalize_actions(action[:self.unnormalize_action_dims])

        return action
    
    def start_episode(self):
        self.policy.start_episode()

    def load_data_stats(self, data_stats_path):
        filename = os.path.expanduser(data_stats_path)
        with open(filename, 'rb') as f:
            data_stats = pickle.load(f)
        return data_stats

class DiffusionRolloutPolicy:

    def __init__(self, policy, action_mean, action_std, stats, policy_config):
        self.policy = policy
        self.policy_config = policy_config
        self.camera_names = policy_config['camera_names']
        self.action_mean = action_mean
        self.action_std = action_std

        self.step = 0
        self.chunk_size = policy_config['num_queries']
        self.cur_action = None

        self.stats = stats
        self.qpos_mean = stats['qpos_mean']
        self.qpos_std = stats['qpos_std']

        # dims to unnormalize - won't unnormalize last len(action) - len(action_mean) dims
        self.unnormalize_dims = len(action_mean)

    def unnormalize(self, action):
        action = ((action + 1)/2) * (self.stats['action_max'] - self.stats['action_min']) + self.stats['action_min']
        action[:self.unnormalize_dims] = action[:self.unnormalize_dims] * self.action_std[:self.unnormalize_dims] + self.action_mean[:self.unnormalize_dims]
        return action

    def preprocess_obs(self, obs):
        qpos = np.concatenate((obs['left'], obs['right'], obs['base'], obs['base_velocity']), axis=-1)
        qpos = (qpos - self.qpos_mean) / self.qpos_std

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(obs[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images).float().cuda().unsqueeze(0)
        qpos_data = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        original_size = image_data.shape[-2:]
        ratio = 0.95
        image_data = image_data[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        image_data = image_data.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        image_data = resize_transform(image_data)
        image_data = image_data.unsqueeze(0)

        return qpos_data, image_data
    
    def generate_action(self, obs):
        with torch.inference_mode():
            qpos, images = self.preprocess_obs(obs)
            action = self.policy(qpos, images)
            
        return action
    
    def get_action(self, obs):
        if self.step == 0:
            # warmup
            for _ in range(10):
                self.generate_action(obs)
            print('network warm up done')

        if self.step % self.chunk_size == 0:
            print('generating new action')
            self.cur_action = self.generate_action(obs)
        # print('cur_action:', self.cur_action.shape)

        action = self.cur_action[:, self.step % self.chunk_size].squeeze(0).cpu().numpy()
        action = self.unnormalize(action)
        self.step += 1
        return action
    
    def start_episode(self):
        self.step = 0


class ACTRolloutPolicy:

    def __init__(self, policy, action_mean, action_std, stats, policy_config):
        self.policy = policy
        self.policy_config = policy_config
        self.camera_names = policy_config['camera_names']
        self.action_mean = action_mean
        self.action_std = action_std

        self.step = 0
        self.chunk_size = policy_config['num_queries']
        self.cur_action = None

        self.qpos_mean = stats['qpos_mean']
        self.qpos_std = stats['qpos_std']

        # dims to unnormalize - won't unnormalize last len(action) - len(action_mean) dims
        self.unnormalize_dims = len(action_mean)

    def unnormalize(self, action):
        action = action * self.action_std + self.action_mean
        return action

    def preprocess_obs(self, obs):
        qpos = np.concatenate((obs['left'], obs['right'], obs['base'], obs['base_velocity']), axis=-1)
        qpos = (qpos - self.qpos_mean) / self.qpos_std


        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(obs[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images).float().cuda().unsqueeze(0)
        qpos_data = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
        return qpos_data, image_data
    
    def generate_action(self, obs):
        with torch.inference_mode():
            qpos, images = self.preprocess_obs(obs)
            action = self.policy(qpos, images)
            
        return action
    
    def get_action(self, obs):
        if self.step == 0:
            # warmup
            for _ in range(10):
                self.generate_action(obs)
            print('network warm up done')

        if self.step % self.chunk_size == 0:
            print('generating new action')
            self.cur_action = self.generate_action(obs)
        # print('cur_action:', self.cur_action.shape)

        action = self.cur_action[:, self.step % self.chunk_size].squeeze(0).cpu().numpy()
        action[:self.unnormalize_dims] = self.unnormalize(action[:self.unnormalize_dims])
        self.step += 1
        return action
    
    def start_episode(self):
        self.step = 0