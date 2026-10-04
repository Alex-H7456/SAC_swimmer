from pathlib import Path

import torch
from torchrl.modules import ProbabilisticActor, TanhNormal
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.modules import MLP, NormalParamExtractor, ValueOperator
from . import SAC_nets
from . import SAC_trainer
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
import logging

logger = logging.getLogger(__name__)
class Agent:
    def __init__(self, cfg, env):
        self.env = env
        self.critic = SAC_nets.basic_sac_critic(cfg, env)
        self.actor = self._process_actor(cfg)
        self.trainer = SAC_trainer.Trainer(
            env, self.actor, self.critic
        )

 
    def _process_actor(self,cfg):
        actor_net = SAC_nets.basic_actor(cfg, self.env)
        actor_extractor = TensorDictModule(
            NormalParamExtractor(
                scale_mapping=f"biased_softplus_{cfg["network"]["default_policy_scale"]}",
                scale_lb=cfg["network"]["scale_lb"],
            ),
            in_keys=["_actor_net_out"],
            out_keys=["loc", "scale"],
        )
        actor_module = TensorDictSequential(actor_net, actor_extractor) # type: ignore

        actor = ProbabilisticActor(
            spec=self.env.action_spec,
            in_keys=["loc", "scale"],
            module=actor_module,
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": self.env.action_spec.space.low,
                "high": self.env.action_spec.space.high,
            },
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )
        return actor
        

    def train(self):
        self.trainer.train()

    def save_actor(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), checkpoint_path)
        logger.info(f"Saved actor checkpoint to {checkpoint_path}", flush=True)

    def load_actor(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.actor.load_state_dict(state_dict)