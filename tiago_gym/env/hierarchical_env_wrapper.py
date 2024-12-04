"""
this wrapper will be used for all environments: serve as a wrapper for hierarchical agent
"""

import gym
from gym import spaces
import numpy as np
import torch
from gym.wrappers import RescaleAction

def to_one_hot(idx, num_classes):
	one_hot = np.zeros(num_classes, dtype=int)
	if idx < 0:
		return one_hot
	one_hot[idx] = 1
	return one_hot

class HierarchicalDiscreteEnv(gym.Env):
	'''
	This env is for multi diayn
	'''
	def __init__(self, env, skill_channel, skill_dim, low_level_steps, device, low_actor=None, vis=False):
		super(HierarchicalDiscreteEnv, self).__init__()

		# This is necessary (since we always apply this in training)
		self._env = RescaleAction(env, min_action=-1, max_action=1)
		self.action_space = spaces.MultiDiscrete([skill_dim] * skill_channel)
		self.observation_space = self._env.observation_space
		self.low_level_actor = low_actor
		self.low_level_steps = low_level_steps  # 50
		self.device = device
		self.skill_channel = skill_channel
		self.skill_dim = skill_dim
		self.vis = vis

	# This function will extract the additional states from the env to generate observations for high-level policy
	def get_full_state(self, obs):
		# Changed from vector to dictionary - we let the get_additional_states take care of
		# full_obs = np.concatenate([obs, self._env.get_additional_states()])
		full_obs = self._env.get_additional_states(obs)
		return full_obs

	def reset(self):
		self.last_observation = self._env.reset()
		return self.get_full_state(self.last_observation.copy())

	def step(self, meta_action):
		reward = 0

		for i in range(self.low_level_steps):
			if self.low_level_actor is None:
				print("low level actor is None")
				action = self._env.action_space.sample()
			else:
				with torch.no_grad():
					obs = torch.as_tensor(self.last_observation, device=self.device, dtype=torch.float32).flatten()
					inputs = [obs]

					skill = np.zeros((self.skill_channel, self.skill_dim), dtype=np.float32)
					skill[range(self.skill_channel), meta_action] = 1.0

					value = torch.as_tensor(skill, device=self.device).flatten()
					inputs.append(value)

					inpt = torch.cat(inputs, dim=-1)
					# We are assumming using SAC
					dist = self.low_level_actor(inpt, 0.2)
					action = dist.mean.cpu().numpy()

			observation, r, done, info = self._env.step(action)
			if self.vis:
				self.render("human")
			reward += r
			if done:
				break
			self.last_observation = observation
		reward /= self.low_level_steps

		full_obs = self.get_full_state(observation)
		# a reward function that only considers the last step
		end_reward = self._env.get_end_skill_reward(full_obs)

		if self._env.factored:
			reward = np.concatenate([reward, [end_reward]])
		else:
			reward += self._env.get_end_skill_reward(full_obs)

		return full_obs, reward, done, info

	def render(self, mode="human"):
		return self._env.render(mode)

	def __getattr__(self, name):
		return getattr(self._env, name)

