# Visualizing Saliency Map of Double-DQN on MoonLander

Based on [this repo](https://github.com/vsaveris/lunar-lander-DQN). Some code detail please check out the original repo. Not the official MoonLander repo since my python environment is broken by Big Sur. The behavior of this agent seems fine though.

# Usage

To simulate, run
```bash
python lunarLander.py -e test -a ./trained_model/DQN_Trained.h5
```
This will generate `output/history.npz`, in which it contains:
- `states`: a matrix of shape `(T, 8)` containing all states that the agent experience, where T is the total number of steps;
- `actions`: a vector of shape `(T,)` containing all actions that the agent took;
- `saliency`: a matrix of shape `(T, 8)` containing the computed saliency for each feature at each step.

To visualize the saliency, run
```bash
python visualize_saliency.py
```
It will read in `output/history.npz` and output `output/saliency.png`, `output/saliency_stacked.png` and `output/saliency.mp4`.
