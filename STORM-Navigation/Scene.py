import sys
import cv2
import h5py
import json
import numpy as np
import random
import torchvision
import gym
import skimage.io
from skimage.transform import resize
import matplotlib.pyplot as plt

ACTION_SIZE = 4
SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
HISTORY_LENGTH = 1


class THORDiscreteEnvironment(object):

    def __init__(self, config=dict()):

        # configurations
        self.scene_name = config.get('scene_name', 'bathroom_03')
        self.random_start = config.get('random_start', True)
        self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1)  # 1 for no sampling
        self.terminal_state_id = config.get('terminal_state_id', 0)

        self.h5_file_path = config.get('h5_file_path', 'data/%s.h5' % self.scene_name)
        self.h5_file = h5py.File(self.h5_file_path, 'r')

        self.locations = self.h5_file['location'][()]
        self.rotations = self.h5_file['rotation'][()]
        self.n_locations = self.locations.shape[0]

        self.terminals = np.zeros(self.n_locations)
        self.terminals[self.terminal_state_id] = 1
        self.terminal_states, = np.where(self.terminals)

        self.transition_graph = self.h5_file['graph'][()]
        self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]

        self.history_length = HISTORY_LENGTH
        self.screen_height = SCREEN_HEIGHT
        self.screen_width = SCREEN_WIDTH

        # we use pre-computed fc7 features from ResNet-50
        # self.s_t = np.zeros([self.screen_height, self.screen_width, self.history_length])
        self.s_t = np.zeros([1, 3, 110, 110])
        self.s_t1 = np.zeros_like(self.s_t)
        self.s_target = self._tiled_state(self.terminal_state_id)

        self.reset()

    # public methods

    def reset(self):
        # randomize initial state
        while True:
            k = random.randrange(self.n_locations)
            min_d = np.inf
            # check if target is reachable
            for t_state in self.terminal_states:
                dist = self.shortest_path_distances[k][t_state]
                min_d = min(min_d, dist)
            # min_d = 0  if k is a terminal state
            # min_d = -1 if no terminal state is reachable from k
            if min_d > 0: break

        # reset parameters
        self.current_state_id = k
        self.s_t = self._tiled_state(self.current_state_id)
        self.s_target = self._tiled_state(self.terminal_state_id)

        self.reward = 0
        self.collided = False
        self.terminal = False
        # self.s_t_reshape = self.s_t.reshape(-1, 2048)
        # self.s_target_reshape = self.s_target.reshape(-1, 2048)

        # return self.s_t_reshape, self.s_target_reshape
        return self.s_t, self.s_target, min_d

    def step(self, action):
        assert not self.terminal, 'step() called in terminal state'
        k = self.current_state_id
        if self.transition_graph[k][action] != -1:
            self.current_state_id = self.transition_graph[k][action]
            if self.terminals[self.current_state_id]:
                self.terminal = True
                self.collided = False
            else:
                self.terminal = False
                self.collided = False
        else:
            self.terminal = False
            self.collided = True

        self.reward = self._reward(self.terminal, self.collided)
       # self.s_t1 = np.append(self.s_t[:, 1:], , axis=1)
        self.state = self._tiled_state(self.current_state_id)
        # self.s_t1_reshape = self.s_t1.reshape(-1, 2048)
        return self.state, self.reward, self.terminal#self.current_state_id, self.collided

    def update(self):
        self.s_t = self.s_t1

    # private methods

    def _tiled_state(self, state_id):

        I = self.h5_file['observation'][state_id]
        # cv2.imwrite('image1.png', I)
       # img_o = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        img_r = cv2.resize(I, (64, 64))
        img_r = img_r[np.newaxis, :, :, :]
      #  img_r = img_r.transpose(2,0,1)
        #img_r = img_r[np.newaxis,:,:,:]
       # f = img_r[:, :, np.newaxis]

        return img_r

        # k = random.randrange(self.n_feat_per_locaiton)
        # f = self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]
        # return np.tile(f, (1, self.history_length))

    def _reward(self, terminal, collided):
        # positive reward upon task completion
        if terminal:
            return 1
        elif collided:
            return -0.1
        else:
            return -0.01
    def random_sample(self):

        action = random.randint(0, 3)
        return action
    # properties

    @property
    def action_size(self):
        # move forward/backward, turn left/right for navigation
        return ACTION_SIZE

    @property
    def action_definitions(self):
        action_vocab = ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]
        return action_vocab[:ACTION_SIZE]

    @property
    def observation(self):
        obs = self.h5_file['observation'][self.current_state_id]
        # return self.h5_file['observation'][self.current_state_id]
        return obs

    @property
    # def state(self):
    #     new_image = self.h5_file['observation'][self.current_state_id]
    #     #img_g = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    #     img_s = cv2.resize(new_image, (64, 64))
    #     #img_s =  img_s.transpose(2,0,1)
    #     img_s = img_s[np.newaxis, :, :, :]
        # scaled new image
       # img_s = img_s[:, :, np.newaxis]
       #  return img_s
        # read from hdf5 cache
        # k = random.randrange(self.n_feat_per_locaiton)
        # return self.h5_file['resnet_feature'][self.current_state_id][k][:, np.newaxis]

    @property
    def target(self):
        return self.s_target

    @property
    def x(self):
        return self.locations[self.current_state_id][0]

    @property
    def z(self):
        return self.locations[self.current_state_id][1]

    @property
    def r(self):
        return self.rotations[self.current_state_id]

    @property
    def observation_space(self):
        spaces = {}
        spaces = gym.spaces.Box(
            0, 255, (3,) + (64, 64), dtype=np.uint8)
        return gym.spaces.Dict({"image": spaces})
    @property
    def action_space(self):
        return gym.spaces.Discrete(4)

# if __name__ == "__main__":
#     scene_name = 'bathroom_02'
#     env = THORDiscreteEnvironment({
#         'random_start': True,
#         'scene_name': scene_name,
#         'h5_file_path': 'data/%s.h5' % scene_name
#     })
    # obs = env.observation()
    # print(type(obs))
