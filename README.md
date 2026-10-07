## Credits
Go to FluidFrame github repo, from which taylor green environment is taken. 
## Goal
Create a SAC swimmer to train a swimmer to navigate vorticity in flow 
## Why 
Use continuous action space of directions unlike previous implementations which use a discrete Q table.

## Train and evaluate

Run training from the repository root:

```bash
venv/bin/python SAC_main.py
```

When training finishes, the actor is saved to
`checkpoints/actor.pt`. The current training run is configured for one million
frames, so it may take a while before this file is created.

Evaluate the saved actor and print episode returns:

```bash
venv/bin/python evaluate.py --checkpoint checkpoints/actor.pt --episodes 5
```

Render the evaluation in Gymnasium's human mode:

```bash
venv/bin/python evaluate.py --checkpoint checkpoints/actor.pt --episodes 1 --render
```

The current environment is `Pendulum-v1`. Its visualisation is the pendulum
window provided by Gymnasium; rendering may require a local graphical session.
The evaluation script uses deterministic actions and reports the return for
each episode. A higher return (closer to zero for Pendulum) indicates better
performance.

The checkpoint contains the actor policy weights only. It is sufficient for
evaluation, but not for resuming the complete optimizer/replay-buffer training
state. An interrupted training process does not currently write the final
actor checkpoint.

Training metrics are also written by TorchRL's CSV logger under
`logs/sac_training/scalars/`. For example,
`logs/sac_training/scalars/r_training.csv` contains step/reward pairs for the
mean per-step reward, while `r_total.csv` contains accumulated episode rewards.
The first column is the number of collected environment frames. Metrics are
logged once per collector batch (currently every 1,000 frames), so these files
can be plotted after training.

### Render a Taylor-Green evaluation

Install the video-rendering dependencies once:

```bash
venv/bin/pip install matplotlib imageio imageio-ffmpeg
```

Then evaluate a Taylor-Green-compatible actor checkpoint and save an MP4:

```bash
venv/bin/python evaluate_taylor_green.py \
  --checkpoint checkpoints/actor.pt \
  --output videos/taylor_green.mp4
```

The video shows fixed Taylor-Green flow contours and the swimmer trajectory.
The default episode lasts 100,000 simulation steps and
the camera is fixed at the initial position with a domain 1.5 times larger
than `--view-size`. The flow field is plotted once at the start and only the
swimmer and trail are updated. Use `--max-steps`, `--fps`, `--seed`, and
`--view-size` to adjust the episode length, playback speed, initial condition,
and plotted domain. The approximate video duration is `max_steps / fps`
seconds; for example,
`--max-steps 1800 --fps 30` produces about one minute of video.
For faster rendering, frames are sampled from the simulation at real-time
frequency by default rather than writing one video frame per simulation step.
Use `--render-interval 1` if you explicitly need every simulation step.
