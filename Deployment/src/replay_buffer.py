import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle


class ReplayBuffer():
    def __init__(self, obs_shape, num_envs, max_length=100000, warmup_length=1024, store_on_gpu=False, device='cuda:6') -> None:
        self.store_on_gpu = store_on_gpu
        self.obs_shape = obs_shape
        self.device = device
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length, num_envs, *obs_shape), dtype=torch.uint8, device=self.device, requires_grad=False)
            self.action_buffer = torch.empty((max_length, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
            self.reward_buffer = torch.empty((max_length, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length, num_envs), dtype=np.float32)
            self.reward_buffer = np.empty((max_length, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length, num_envs), dtype=np.float32)

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def load_trajectory(self, path):
        buffer = pickle.load(open(path, "rb"))
        if self.store_on_gpu:
            self.external_buffer = {name: torch.from_numpy(buffer[name]).to(self.device) for name in buffer}
        else:
            self.external_buffer = buffer
        self.external_buffer_length = self.external_buffer["obs"].shape[0]

    def sample_external(self, batch_size, batch_length, to_device="cuda"):
        indexes = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
        if self.store_on_gpu:
            obs = torch.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = torch.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = torch.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = torch.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        else:
            obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        return obs, action, reward, termination

    def ready(self):
        return self.length > self.warmup_length

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length, to_device="cuda"):
        if self.store_on_gpu:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size)
                    obs.append(torch.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(torch.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(torch.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(torch.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length, self.device)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.cat(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)

        return obs, action, reward, termination

    def append(self, obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % (self.max_length)

        self.obs_buffer[self.last_pointer] = obs
        self.reward_buffer[self.last_pointer, :] = reward
        self.action_buffer[self.last_pointer] = action.squeeze()

        self.termination_buffer[self.last_pointer] = termination.squeeze()

        if self.length < self.max_length:
            self.length += 1


