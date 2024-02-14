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