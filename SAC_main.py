from env.test_env import make_env
from model.agent import Agent
from env.taylor_green_gym import make_taylor_green_continuous_env
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

def build_basis_agent():
    if not cfg["solver"]["taylor_green"]:
            env = make_env() #Using test gym environment for inverted pendulum 
    else:
        print("Using closed-form analytical solution ...")
        env = make_taylor_green_continuous_env(
            dt=0.01,
            swimmer_speed= cfg["solver"]["swimmer_speed"],
            alignment_timescale= cfg["solver"]["alignment_timescale"],
            seed=42,
            action_type="continuous",
        )  # initialise environment

    return Agent(cfg,env), env


def main() -> None:
    agent, env = build_basis_agent()
    logger.info(f"Environment: {env} \n")
    logger.info(f"Observation spec: {env.observation_spec} \n")
    logger.info(f"Action spec: {env.action_spec} \n ")

    agent.train()
    agent.save_actor(CHECKPOINT_PATH)
    env.close()


if __name__ == "__main__":
    main()
