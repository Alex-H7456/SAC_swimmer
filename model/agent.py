from torchrl.modules import ProbabilisticActor, TanhNormal

from . import SAC_nets
from . import SAC_trainer


class Agent:
    def __init__(self, cfg, env):
        self.env = env
        self.actor = SAC_nets.basic_actor(cfg, self.env.action_spec)
        self.critic = SAC_nets.basic_sac_critic(cfg)
        self.policy = self._get_policy()
        self.trainer = SAC_trainer.Trainer(
            env, self.policy, self.actor, self.critic
        )

    def _get_policy(self):
        policy = ProbabilisticActor(
            module=self.actor,
            spec=self.env.action_spec,
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": self.env.action_spec.low,
                "max": self.env.action_spec.high,
            },
            in_keys=["loc", "scale"],
            out_keys=["action"],
            return_log_prob=True,
        )
        return policy

    def train(self):
        self.trainer.train()