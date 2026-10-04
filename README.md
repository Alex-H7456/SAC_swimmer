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