import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tensordict.nn import InteractionType, set_interaction_type
from tqdm import tqdm

from env.taylor_green_gym import make_taylor_green_continuous_env
from model.agent import Agent


def _initialise_plot(
    figure: Figure,
    axis: Axes,
    position: np.ndarray,
    trail: list[np.ndarray],
    view_size: float,
    flow_speed: float,
) -> tuple[object, object, object]:
    half_view = 3.0 * view_size / 2
    flow_axis = np.linspace(position[0] - half_view, position[0] + half_view, 14)
    flow_axis_y = np.linspace(position[1] - half_view, position[1] + half_view, 14)
    flow_x, flow_y = np.meshgrid(flow_axis, flow_axis_y)
    stream_function = 0.5 * flow_speed * np.sin(flow_x) * np.sin(flow_y)
    axis.contour(
        flow_x,
        flow_y,
        stream_function,
        levels=20,
        cmap="Blues",
        linewidths=0.8,
        alpha=0.8,
    )
    trail_array = np.asarray(trail)
    trail_artist, = axis.plot(
        trail_array[:, 0], trail_array[:, 1], color="tab:orange", linewidth=1.5
    )
    swimmer_artist = axis.scatter(
        [position[0]], [position[1]], color="tab:red", s=55, zorder=3
    )
    title_artist = axis.set_title("Taylor-Green swimmer | step 0 | return 0.000")
    axis.set(
        xlim=(position[0] - half_view, position[0] + half_view),
        ylim=(position[1] - half_view, position[1] + half_view),
        aspect="equal",
        xlabel="x",
        ylabel="y",
    )
    return trail_artist, swimmer_artist, title_artist


def _make_frame(
    figure: Figure,
    position: np.ndarray,
    trail: list[np.ndarray],
    episode_return: float,
    step: int,
    artists: tuple[object, object, object],
) -> np.ndarray:
    trail_artist, swimmer_artist, title_artist = artists
    trail_array = np.asarray(trail)
    trail_artist.set_data(trail_array[:, 0], trail_array[:, 1])
    swimmer_artist.set_offsets([position])
    title_artist.set_text(
        f"Taylor-Green swimmer | step {step} | return {episode_return:.3f}"
    )
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def _write_episode_video(
    agent: Agent,
    env,
    output_path: Path,
    max_steps: int,
    fps: int,
    view_size: float,
    render_interval: int,
) -> float:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video output requires imageio and imageio-ffmpeg. "
            "Install them with: pip install imageio imageio-ffmpeg"
        ) from exc

    # GymWrapper retains the custom Gym environment at this stable public wrapper boundary.
    gym_env = env.base_env.unwrapped
    simulation = gym_env._env
    flow_speed = simulation.u0
    tensordict = env.reset()
    position = simulation.swimmer_position.copy()
    trail = [position.copy()]
    episode_return = 0.0
    figure, axis = plt.subplots(figsize=(7.04, 7.04), dpi=100)
    artists = _initialise_plot(
        figure, axis, position, trail, view_size, flow_speed
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
        ffmpeg_params=[
            "-profile:v",
            "baseline",
            "-level",
            "4.0",
            "-movflags",
            "+faststart",
        ],
    ) as writer:
        writer.append_data(
            _make_frame(
                figure, position, trail, episode_return, 0, artists
            )
        )
        with set_interaction_type(InteractionType.DETERMINISTIC), torch.no_grad():
            for step in tqdm(range(1, max_steps + 1)):
                tensordict = agent.actor(tensordict)
                tensordict = env.step(tensordict)
                episode_return += tensordict["next", "reward"].item()
                tensordict = tensordict["next"]

                position = simulation.swimmer_position.copy()
                trail.append(position.copy())
                if step % render_interval == 0 or step == max_steps:
                    writer.append_data(
                        _make_frame(
                            figure,
                            position,
                            trail,
                            episode_return,
                            step,
                            artists,
                        )
                    )
    plt.close(figure)
    return episode_return


def evaluate(
    checkpoint: str,
    output: str,
    max_steps: int,
    fps: int,
    seed: int,
    view_size: float,
    render_interval: int | None,
) -> None:
    with open("config.yaml", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    swimmer_speed = cfg["solver"]["swimmer_speed"]
    alignment_timescale = cfg["solver"]["alignment_timescale"]
    env = make_taylor_green_continuous_env(
        dt=0.01,
        swimmer_speed=swimmer_speed,
        alignment_timescale=alignment_timescale,
        seed=seed,
        action_type="continuous",
        max_episode_steps=max_steps,
    )
    agent = Agent(cfg, env)
    agent.load_actor(checkpoint)
    try:
        if render_interval is None:
            render_interval = max(1, round(1.0 / (0.01 * fps)))
        episode_return = _write_episode_video(
            agent,
            env,
            Path(output),
            max_steps,
            fps,
            view_size=view_size,
            render_interval=render_interval,
        )
        print(f"Saved {output} (return={episode_return:.3f})", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Taylor-Green SAC actor and save a rendered MP4."
    )
    parser.add_argument("--checkpoint", default="checkpoints/actor.pt")
    parser.add_argument("--output", default="videos/taylor_green.mp4")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--render-interval",
        type=int,
        default=None,
        help="Simulation steps between video frames; defaults to real-time sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--view-size",
        type=float,
        default=2 * np.pi,
        help="Width and height of the camera window around the swimmer.",
    )
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.output,
        args.max_steps,
        args.fps,
        args.seed,
        args.view_size,
        args.render_interval,
    )
