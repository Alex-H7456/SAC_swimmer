from math import prod

from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import MLP, NormalParamExtractor, ValueOperator


def _feature_count(spec):
    return prod(spec.shape) or 1


def basic_actor(cfg, env, in_keys=["observation"], out_keys = ["_actor_net_out"]):
    observation_spec = env.observation_spec["observation"]
    action_spec = env.action_spec
    actor_module = TensorDictModule(
        MLP(num_cells=cfg["network"]["actor_hidden_sizes"],
            in_features=_feature_count(observation_spec),
            out_features=2 * _feature_count(action_spec),
            activation_class=nn.ReLU),
        in_keys=in_keys,
        out_keys=out_keys,
    )

    return actor_module


def basic_sac_critic(cfg, env, in_keys=["observation"]):
    observation_spec = env.observation_spec["observation"]
    action_spec = env.action_spec
    qvalue_net_kwargs = {
        "num_cells": cfg["network"]["sac_critic_hidden_sizes"],
        "in_features": _feature_count(observation_spec) + _feature_count(action_spec),
        "out_features": 1,
        "activation_class": nn.ReLU,
    }
    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )
    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    return qvalue
