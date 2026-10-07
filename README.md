# SAC swimmer in Taylor–Green flow

This project trains a swimmer to navigate a Taylor–Green vortex using **Soft
Actor-Critic (SAC)** with a continuous action space. The policy can therefore
choose continuous swimming directions rather than selecting from a discrete
action table.

The neural networks are implemented in **PyTorch** and integrated with
TorchRL. The Taylor–Green environment uses a closed-form analytical solution
for the flow field. After training, the learned actor can be evaluated and
rendered as an MP4 showing the flow contours and swimmer trajectory.

## Project structure

- `SAC_main.py` — loads the configuration, creates the selected environment,
  trains SAC, and saves the actor.
- `config.yaml` — network, solver, and Taylor–Green environment parameters.
- `env/` — Taylor–Green and test environments.
- `model/` — the PyTorch SAC actor, critic, and trainer.
- `evaluate_taylor_green.py` — evaluates an actor and writes a Taylor–Green
  video.
- `evaluate_pendulum.py` — evaluates the optional Pendulum test environment.

## Installation

Create a virtual environment and install the dependencies:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Video generation requires Matplotlib and ImageIO's FFmpeg backend. Install
them if they are not already available:

```bash
venv/bin/pip install matplotlib imageio imageio-ffmpeg
```

## Configuration

The default configuration in `config.yaml` selects the continuous Taylor–Green
environment:

```yaml
solver:
  taylor_green: True
```

The same file controls the actor and critic hidden layers, flow and swimmer
parameters, integration time step, diffusion coefficients, and maximum episode
length. Edit the file before training to change these settings.

## Training

Run training from the repository root:

```bash
venv/bin/python SAC_main.py
```

The trainer currently runs for one million environment frames. When training
finishes, the actor weights are saved to:

```text
checkpoints/actor.pt
```

Training metrics are written as CSV files under
`logs/sac_training/scalars/`. The checkpoint contains the actor policy weights
for evaluation; it does not contain the optimizer or replay-buffer state
needed to resume training.

## Taylor–Green evaluation and video

Evaluate the trained actor and save an MP4:

```bash
venv/bin/python evaluate_taylor_green.py \
  --checkpoint checkpoints/actor.pt \
  --output videos/taylor_green.mp4
```

The default evaluation runs for 100,000 simulation steps at 60 frames per
second. For a shorter video, for example:

```bash
venv/bin/python evaluate_taylor_green.py \
  --checkpoint checkpoints/actor.pt \
  --output videos/taylor_green_short.mp4 \
  --max-steps 1800 \
  --fps 30
```

Useful options include:

- `--max-steps` — number of simulation steps.
- `--fps` — output video frame rate.
- `--seed` — evaluation seed.
- `--view-size` — width and height of the plotted camera window.
- `--render-interval` — simulation steps between video frames. By default,
  frames are sampled at approximately real-time frequency; use
  `--render-interval 1` to write every simulation step.

The command prints the final episode return after saving the video. The
resulting animation contains fixed Taylor–Green flow contours, the swimmer,
and its trajectory.

## Pendulum test environment

The Gymnasium Pendulum environment is included as an alternative test
environment for checking the SAC network and training pipeline independently
of the Taylor–Green swimmer.

Set the solver flag in `config.yaml` to:

```yaml
solver:
  taylor_green: False
```

Then train as usual:

```bash
venv/bin/python SAC_main.py
```

To evaluate the resulting actor, optionally use Gymnasium's human rendering:

```bash
venv/bin/python evaluate_pendulum.py \
  --checkpoint checkpoints/actor.pt \
  --render
```

## Credit

The Taylor–Green environment is based on the
[FluidFrame repository](https://github.com/sm7610/fluidframe). Many thanks to
the authors and contributors for making that work available.
