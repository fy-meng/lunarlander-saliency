# Visualizing Saliency Map of Double-DQN on MoonLander

Based on [this repo](https://github.com/vsaveris/lunar-lander-DQN). Some code detail please check out the original repo. Not the official MoonLander repo since my python environment is broken by Big Sur. The behavior of this agent seems fine though.

# Usage

To simulate, run
```bash
lunarLander.py [-h] [-v {0,1,2}] -e {train,test} [-a A] [-n N]
               [--rendering] [--store_history] [--compute_saliency]

```

For example, to run 10 trials and store history, including saliency:
```bash
python lunarLander.py -e test -a ./trained_model/DQN_Trained.h5 -n 10 --save_history --compute_saliency
```
This will generate `output/history.npz`, in which it contains:
- `trial`: a vector of shape `(T,)` containing the trial number for each data point, where T is the total number of steps;
- `state`: a matrix of shape `(T, 8)` containing states at each step;
- `action`: a vector of shape `(T,)` containing actions at each step;
- `reward`: a vector of shape `(T,)` containing reward at each step;
- `next_state`: a matrix of shape `(T, 8)` containing the next state at each step;
- `q_values`: a matrix of shape `(T, 4)` containig the Q-values at each step;
- `saliency`: a matrix of shape `(T, 8)` containing the computed saliency for each feature at each step.
Note that the `saliency` only exists if the `--compute_saliency` tag is used.

To visualize the saliency, run
```bash
python visualize_saliency.py
```
It will read in `output/history.npz` and output `output/saliency.png`, `output/saliency_stacked.png` and `output/saliency.mp4`.
