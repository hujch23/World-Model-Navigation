import torch
import yaml
import torchvision.transforms as transforms
from src.policy import NavNetPolicy
from typing import Any, Dict, List, Optional, Tuple
from tensordict import (
    DictTree,
    TensorDict,
    TensorOrNDArrayDict,
)
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numbers
from collections import OrderedDict
import numpy as np
import os
import pickle


class NeuralNetNode:
    def __init__(self):
        with open('config_of_eval.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cpu")
        self._pool: Dict[Any, Union[torch.Tensor, np.ndarray]] = {}
        self.actor_critic = self._setup_actor_critic_agent()

        self.actor_critic = torch.load('model.pth', map_location="cpu")
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            2,
            256,
            device=self.device,
        )
        self.prev_actions = torch.zeros(
            1,
            1,
            device=self.device,
            dtype=torch.long,
        )
        self.not_done_masks = torch.ones(
            1,
            1,
            device=self.device,
            dtype=torch.bool,
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def get(
        self,
        num_obs: int,
        sensor_name: Any,
        sensor: torch.Tensor,
        device: Optional[torch.device] = None,
    ):
        r"""Returns a tensor of the right size to batch num_obs observations together

        If sensor is a cpu-side tensor and device is a cuda device the batched tensor will
        be pinned to cuda memory.  If sensor is a cuda tensor, the batched tensor will also be
        a cuda tensor
        """

        key = (
            sensor_name,
            tuple(sensor.size()),
            sensor.type(),
            sensor.device.type,
            sensor.device.index,
        )
        if key in self._pool:
            cache = self._pool[key]
            if cache.shape[0] >= num_obs:
                return cache[0:num_obs]
            else:
                cache = None
                del self._pool[key]

        cache = torch.empty(
            num_obs, *sensor.size(), dtype=sensor.dtype, device=sensor.device
        )
        if (
            device is not None
            and device.type == "cuda"
            and cache.device.type == "cpu"
        ):
            cache = cache.pin_memory()

        if cache.device.type == "cpu":
            # Pytorch indexing is slow,
            # so convert to numpy
            cache = cache.numpy()

        self._pool[key] = cache
        return cache


    def batch_obs(
        self,
        observations: List[DictTree],
        device: Optional[torch.device] = None,
    ) -> TensorDict:
        observations = [
            TensorOrNDArrayDict.from_tree(o).map(
                lambda t: t.numpy()
                if isinstance(t, torch.Tensor) and t.device.type == "cpu"
                else t
            )
            for o in observations
        ]
        observation_keys, _ = observations[0].flatten()
        observation_tensors = [o.flatten()[1] for o in observations]

        # Order sensors by size, stack and move the largest first
        upload_ordering = sorted(
            range(len(observation_keys)),
            key=lambda idx: 1
            if isinstance(observation_tensors[0][idx], numbers.Number)
            else int(np.prod(observation_tensors[0][idx].shape)),  # type: ignore
            reverse=True,
        )

        batched_tensors = []
        for sensor_name, obs in zip(observation_keys, observation_tensors[0]):
            batched_tensors.append(
                self.get(
                    len(observations),
                    sensor_name,
                    torch.as_tensor(obs),
                    device,
                )
            )

        for idx in upload_ordering:
            for i, all_obs in enumerate(observation_tensors):
                obs = all_obs[idx]
                if isinstance(obs, np.ndarray):
                    batched_tensors[idx][i] = obs  # type: ignore
                elif isinstance(obs, torch.Tensor):
                    batched_tensors[idx][i].copy_(obs, non_blocking=True)  # type: ignore
                # If the sensor wasn't a tensor, then it's some CPU side data
                # so use a numpy array
                else:
                    batched_tensors[idx][i] = np.asarray(obs)  # type: ignore


            if isinstance(batched_tensors[idx], np.ndarray):
                batched_tensors[idx] = torch.from_numpy(batched_tensors[idx])

            batched_tensors[idx] = batched_tensors[idx].to(  # type: ignore
                device, non_blocking=True
            )

        return TensorDict.from_flattened(observation_keys, batched_tensors)




    def _setup_actor_critic_agent(self) -> None:

        from gym import spaces

        # 定义 obs_space
        self.observation_space = spaces.Dict({
            'imagegoal_sensor_v2': spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            'rgb': spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
        })

        self.policy_action_space = spaces.Discrete(4)

        self.actor_critic = NavNetPolicy.from_config(
            self.config,
            self.observation_space,
            self.policy_action_space,
            orig_action_space=4,
        )

        return self.actor_critic

    def save_spaces_to_torch(self, observation_space, policy_action_space, orig_policy_action_space, file_path='spaces.pt'):
        torch.save({
            'observation_space': observation_space,
            'policy_action_space': policy_action_space,
            'orig_policy_action_space': orig_policy_action_space,
        }, file_path)

    def load_spaces_from_torch(file_path='spaces.pt'):
        with open('spaces.pt', 'rb') as f:
            loaded_data = torch.load(f)
        return loaded_data['observation_space'], loaded_data['policy_action_space'], loaded_data['orig_policy_action_space']
    
    def inference(self, observations):
        r"""Set the model to inference mode.
        """
        batch = self.batch_obs(observations, device=self.device)
        _, actions, action_log_probs, test_recurrent_hidden_states = self.actor_critic.act(batch, self.test_recurrent_hidden_states, self.prev_actions, self.not_done_masks, deterministic=False)
        self.test_recurrent_hidden_states = test_recurrent_hidden_states
        self.prev_actions = actions

        return actions

if __name__ == '__main__':
    node = NeuralNetNode()
    img_goal = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)  # 小车相机图像
    img_obs = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)  # 读取目标图像
    obs_goal = OrderedDict([('imagegoal_sensor_v2', img_goal), ('rgb', img_obs)])
    observations = [obs_goal]
    actions = node.inference(observations)
    print(actions)



        




