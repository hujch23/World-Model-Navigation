#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from typing import Dict, Iterable, Optional, Union
from torch import Size, Tensor
import torch
from gym import spaces
from torch import nn as nn
import math

from gym import Space


class EmptySpace(Space):
    """
    A ``gym.Space`` that reflects arguments space for action that doesn't have
    arguments. Needed for consistency ang always samples `None` value.
    """

    def sample(self):
        return None

    def contains(self, x):
        if x is None:
            return True
        return False

    def __repr__(self):
        return "EmptySpace()"
def iterate_action_space_recursively(action_space):
    if isinstance(action_space, spaces.Dict):
        for v in action_space.values():
            yield from iterate_action_space_recursively(v)
    else:
        yield action_space

def get_num_actions(action_space) -> int:
    num_actions = 0
    for v in iterate_action_space_recursively(action_space):
        if isinstance(v, spaces.Box):
            assert (
                len(v.shape) == 1
            ), f"shape was {v.shape} but was expecting a 1D action"
            num_actions += v.shape[0]
        elif isinstance(v, EmptySpace):
            num_actions += 1
        elif isinstance(v, spaces.Discrete):
            num_actions += v.n
        else:
            raise NotImplementedError(
                f"Trying to count the number of actions with an unknown action space {v}"
            )

    return num_actions


class CustomFixedCategorical(torch.distributions.Categorical):  # type: ignore
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: Tensor) -> Tensor:
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1, keepdim=True)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)

class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: Tensor) -> CustomFixedCategorical:
        x = self.linear(x)
        return CustomFixedCategorical(logits=x.float(), validate_args=False)


class CustomNormal(torch.distributions.normal.Normal):
    def sample(
        self, sample_shape: Size = torch.Size()  # noqa: B008
    ) -> Tensor:
        return self.rsample(sample_shape)

    def log_probs(self, actions) -> Tensor:
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self) -> Tensor:
        return super().entropy().sum(-1, keepdim=True)


class GaussianNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        config,
    ) -> None:
        super().__init__()

        self.action_activation = config.action_activation
        self.use_softplus = config.use_softplus
        self.use_log_std = config.use_log_std
        use_std_param = config.use_std_param
        self.clamp_std = config.clamp_std

        if self.use_log_std:
            self.min_std = config.min_log_std
            self.max_std = config.max_log_std
            std_init = config.log_std_init
        elif self.use_softplus:
            inv_softplus = lambda x: math.log(math.exp(x) - 1)
            self.min_std = inv_softplus(config.min_std)
            self.max_std = inv_softplus(config.max_std)
            std_init = inv_softplus(1.0)
        else:
            self.min_std = config.min_std
            self.max_std = config.max_std
            std_init = 1.0  # initialize std value so that std ~ 1

        if use_std_param:
            self.std = torch.nn.parameter.Parameter(
                torch.randn(num_outputs) * 0.01 + std_init
            )
            num_linear_outputs = num_outputs
        else:
            self.std = None
            num_linear_outputs = 2 * num_outputs

        self.mu_maybe_std = nn.Linear(num_inputs, num_linear_outputs)
        nn.init.orthogonal_(self.mu_maybe_std.weight, gain=0.01)
        nn.init.constant_(self.mu_maybe_std.bias, 0)

        if not use_std_param:
            nn.init.constant_(self.mu_maybe_std.bias[num_outputs:], std_init)

    def forward(self, x: Tensor) -> CustomNormal:
        mu_maybe_std = self.mu_maybe_std(x).float()
        if self.std is not None:
            mu = mu_maybe_std
            std = self.std
        else:
            mu, std = torch.chunk(mu_maybe_std, 2, -1)

        if self.action_activation == "tanh":
            mu = torch.tanh(mu)

        if self.clamp_std:
            std = torch.clamp(std, self.min_std, self.max_std)
        if self.use_log_std:
            std = torch.exp(std)
        if self.use_softplus:
            std = torch.nn.functional.softplus(std)

        return CustomNormal(mu, std, validate_args=False)






class Policy(abc.ABC):
    action_distribution: nn.Module

    def __init__(self):
        pass

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return 0

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass


class NetPolicy(nn.Module, Policy):
    aux_loss_modules: nn.ModuleDict

    def __init__(
        self, net, action_space, policy_config=None, aux_loss_config=None
    ):
        super().__init__()
        self.net = net
        self.dim_actions = get_num_actions(action_space)
        self.action_distribution: Union[CategoricalNet, GaussianNet]

        if policy_config is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )

        if self.action_distribution_type == "categorical":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif self.action_distribution_type == "gaussian":
            self.action_distribution = GaussianNet(
                self.net.output_size,
                self.dim_actions,
                policy_config.action_dist,
            )
        else:
            ValueError(
                f"Action distribution {self.action_distribution_type}"
                "not supported."
            )

        self.critic = CriticHead(self.net.output_size)

        self.aux_loss_modules = nn.ModuleDict()
        for aux_loss_name in (
            () if aux_loss_config is None else aux_loss_config.enabled
        ):
            # aux_loss = baseline_registry.get_auxiliary_loss(aux_loss_name)

            self.aux_loss_modules[aux_loss_name] = aux_loss(
                action_space,
                self.net,
                **getattr(aux_loss_config, aux_loss_name),
            )

    @property
    def should_load_agent_state(self):
        return True

    @property
    def num_recurrent_layers(self) -> int:
        return self.net.num_recurrent_layers

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        # features, rnn_hidden_states, _ = self.net(
        #     observations, rnn_hidden_states, prev_actions, masks
        # )
        features = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )


        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        # return value, action, action_log_probs, rnn_hidden_states
        return value, action, action_log_probs


    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info: Dict[str, torch.Tensor],
    ):
        features, rnn_hidden_states, action_loss = self.net(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            rnn_build_seq_info,
            is_train=True,
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        batch = dict(
            observations=observations,
            rnn_hidden_states=rnn_hidden_states,
            prev_actions=prev_actions,
            masks=masks,
            action=action,
            rnn_build_seq_info=rnn_build_seq_info,
        )
        aux_loss_res = {
            k: v(aux_loss_state, batch)
            for k, v in self.aux_loss_modules.items()
        }

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_loss_res,
            action_loss,
        )

    @property
    def policy_components(self):
        return (self.net, self.critic, self.action_distribution)

    def policy_parameters(self) -> Iterable[torch.Tensor]:
        for c in self.policy_components:
            yield from c.parameters()

    def all_policy_tensors(self) -> Iterable[torch.Tensor]:
        yield from self.policy_parameters()
        for c in self.policy_components:
            yield from c.buffers()

    def aux_loss_parameters(self) -> Dict[str, Iterable[torch.Tensor]]:
        return {k: v.parameters() for k, v in self.aux_loss_modules.items()}

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        pass


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


# @baseline_registry.register_policy
class PointNavBaselinePolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        aux_loss_config=None,
        **kwargs,
    ):
        super().__init__(
            PointNavBaselineNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space=action_space,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config,
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass

    @property
    @abc.abstractmethod
    def perception_embedding_size(self) -> int:
        pass
