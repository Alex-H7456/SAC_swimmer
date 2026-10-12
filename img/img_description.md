### `q_learning_curve.png`

Learning curve for tabular Q-learning over environment timesteps using 8 seeds, 100 episodes per seed, 10,000 steps per episode, `dt=0.01`, swimmer speed `0.3`, flow speed `1.0`, alignment timescale `1.0`, rotational diffusivity `0.0001`, translational diffusivity `0.001`, `gamma=0.999`, learning rate `0.1`, and epsilon annealed from `0.3` to `0.05`. The tabular agent receives only a 12-state encoding of vorticity band and dominant swimming direction, not absolute position, so this observation is a partial view of the physical state and the learning problem is a POMDP.

### `benchmark_learning_curve.png`

Learning curves compare SAC, tabular Q-learning, and the naive swimmer over environment timesteps using the benchmark defaults of 4 seeds, 100 episodes per seed, 10,000 steps per episode, `dt=0.01`, swimmer speed `0.3`, flow speed `1.0`, alignment timescale `1.0`, and the same diffusivities listed above, with individual seed traces and percentile bands showing the distributions. The naive swimmer always swims upward, and this figure is from the FluidFrame repository referenced in the README.

### `example.png`

The figure shows the SAC swimmer navigating through a Taylor–Green vortex flow, with its trajectory overlaid on flow contours. The SAC agent uses continuous observations and actions to control its swimming direction.
