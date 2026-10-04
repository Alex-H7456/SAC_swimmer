from env.test_env import make_env
from model.agent import Agent
from torchrl.envs import GymEnv
import yaml 

CHECKPOINT_PATH = "checkpoints/actor.pt"

with open("config.yaml") as stream:
    cfg = yaml.safe_load(stream)

def build_agent() -> tuple[Agent, GymEnv]:
    """Create the environment and connect it to the SAC agent."""
    env = make_env()
    agent = Agent(cfg, env)
    return agent, env


def main() -> None:
    agent, env = build_agent()
    try:
        print(f"Environment: {env}")
        print(f"Observation spec: {env.observation_spec}")
        print(f"Action spec: {env.action_spec}")
        agent.train()
        agent.save_actor(CHECKPOINT_PATH)
    finally:
        env.close()


if __name__ == "__main__":
    main()
