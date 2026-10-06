import argparse

import torch
from tensordict.nn import InteractionType, set_interaction_type

from env.test_env import make_env
from SAC_main import build_basis_agent


def evaluate(checkpoint: str, episodes: int, render: bool) -> None:
    train_agent, train_env = build_basis_agent()
    train_env.close()
    train_agent.load_actor(checkpoint)

    env = make_env("human" if render else None)
    try:
        returns = []
        with set_interaction_type(InteractionType.DETERMINISTIC):
            for episode in range(episodes):
                tensordict = env.reset()
                episode_return = 0.0
                done = False
                while not done:
                    tensordict = train_agent.actor(tensordict)
                    tensordict = env.step(tensordict)
                    episode_return += tensordict["next", "reward"].item()
                    done = tensordict["next", "done"].item()
                    tensordict = tensordict["next"]
                returns.append(episode_return)
                print(f"Episode {episode + 1}: return={episode_return:.2f}", flush=True)
        print(f"Mean return: {torch.tensor(returns).mean().item():.2f}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and optionally render a trained SAC actor.")
    parser.add_argument("--checkpoint", default="checkpoints/actor.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.episodes, args.render)
