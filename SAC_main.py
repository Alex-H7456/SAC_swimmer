from env.test_env import make_env
from model.agent import Agent
from torchrl.envs import GymEnv
import yaml 
import logging 

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Custom time format
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Environment: {env} \n")
    logger.info(f"Observation spec: {env.observation_spec} \n")
    logger.info(f"Action spec: {env.action_spec} \n ")
    agent.train()
    agent.save_actor(CHECKPOINT_PATH)
    env.close()


if __name__ == "__main__":
    main()
