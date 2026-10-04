from dataclasses import dataclass
from types import SimpleNamespace

from torchrl.envs import GymEnv, RewardSum, TransformedEnv

ENV_ID = "Pendulum-v1"


def make_env(render_mode: str | None = None) -> TransformedEnv:
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    return TransformedEnv(
        GymEnv(ENV_ID, **kwargs),
        RewardSum(
            in_keys=["reward"],
            out_keys=["reward_sum"],
        ),
    )
