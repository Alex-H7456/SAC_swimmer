from dataclasses import dataclass
from types import SimpleNamespace

from torchrl.envs import GymEnv

ENV_ID = "Pendulum-v1"


@dataclass
class NetworkConfig:
    actor_hidden_sizes: tuple[int, ...] = (256, 256)
    sac_critic_hidden_sizes: tuple[int, ...] = (256, 256)


def make_env(render_mode: str | None = None) -> GymEnv:
    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    return GymEnv(ENV_ID, **kwargs)


def make_config() -> SimpleNamespace:
    return SimpleNamespace(network=NetworkConfig())