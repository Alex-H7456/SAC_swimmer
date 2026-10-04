import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import MLP, ValueOperator
from typing import Tuple


def basic_actor(cfg, action_spec, in_keys=["observation"], out_keys=["action"]):
    
    activation = nn.ReLU

    actor_module = TensorDictModule(
        MLP(num_cells=cfg.network.actor_hidden_sizes,
            out_features=2 * action_spec.shape[-1],
            activation_class=activation),
        in_keys=in_keys,
        out_keys=out_keys,
    )

    return actor_module


def basic_sac_critic(cfg, in_keys=["observation"]):
    qvalue_net_kwargs = {
        "num_cells": cfg.network.sac_critic_hidden_sizes,
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
