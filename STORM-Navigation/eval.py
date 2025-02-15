import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os
from Scene import THORDiscreteEnvironment as Environment
from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp
from sub_models.world_models import WorldModel, MSELoss


def process_visualize(img):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (640, 640))
    return img


def build_single_env(env_name, image_size):
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    return env


def build_vec_env(env_name, image_size, num_envs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def eval_episodes(num_episode, env_name, max_steps, num_envs, image_size,
                  world_model: WorldModel, agent: agents.ActorCriticAgent):
    world_model.eval()
    agent.eval()
    vec_env = Environment({'scene_name': 'bathroom_02', 'terminal_state_id': 26})
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    sum_reward = 0
    current_obs, mid = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    l = 0
    epi_length = []
    min_length = []
    # for total_steps in tqdm(range(max_steps//num_envs)):
    for i in range(num_episode):
        # sample part >>>
        min_length.append(mid)
        for step in range(500):

            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.random_sample()
                    action = np.array(action).reshape(1, )
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )

            context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
            context_action.append(action)
            action = np.squeeze(action)

            obs, reward, done = vec_env.step(action)
            # cv2.imshow("current_obs", process_visualize(obs[0]))
            # cv2.waitKey(10)
            sum_reward += reward
            current_obs = obs
            l = l + 1


            if done:
                context_obs.clear()
                context_action.clear()
                sum_reward = 0
                epi_length.append(l)
                l = 0
                current_obs, mid = vec_env.reset()
                break
    num_fail = 0
    for jj in range(100):
        if epi_length[jj] == 500:
            num_fail = num_fail + 1
    SR = 1 - num_fail / 100

    SPL = 0
    for ii in range(100):
        if epi_length[ii] != 500:
            SPL = SPL + min_length[ii] / max(epi_length[ii], min_length[ii])
    SPL = SPL / 100 * 100


    return SR, SPL

        # update current_obs, current_info and sum_reward

        # <<< sample part


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", type=str, default='config_files/STORM.yaml')
    parser.add_argument("-env_name", type=str, default='DreamerNavigation')
    parser.add_argument("-run_name", type=str, default='DreamerNavigation-100k-seed1')

    args = parser.parse_args()
    conf = load_config(args.config_path)
    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    # print(colorama.Fore.RED + str(conf) + colorama.Style.RESET_ALL)

    # set seed
#    seed_np_torch(seed=conf.BasicSettings.Seed)

    # build and load model/agent
    import train
    #dummy_env = build_single_env(args.env_name, conf.BasicSettings.ImageSize)

    action_dim = 4
    #vec_env = Environment({'scene_name': 'bathroom_02', 'terminal_state_id': 26})

    world_model = train.build_world_model(conf, action_dim)
    agent = train.build_agent(conf, action_dim)
    root_path = f"ckpt/{args.run_name}"

    import glob
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[-1:]
    print(steps)
    results = []
    for step in tqdm(steps):

        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth"))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth"))
        # # eval
        SR, SPL = eval_episodes(
            num_episode=100,
            env_name=args.env_name,
            num_envs=1,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            image_size=conf.BasicSettings.ImageSize,
            world_model=world_model,
            agent=agent
        )
        print('SR:', SR * 100)
        print('SPL:', SPL)
        #print(episode_return)
       # results.append(episode_return)

    # with open(f"eval_result/{args.run_name}.csv", "w") as fout:
    #     fout.write("step, episode_avg_return\n")
    #     for step, episode_avg_return in results:
    #         fout.write(f"{step},{episode_avg_return}\n")
