import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

state_labels = [
    'x_pos', 'y_pos', 'x_vel', 'y_vel',
    'angle', 'ang_vel', 'left_leg', 'right_leg'
]

# load data
file = 'output/history.npz'
data = np.load(file)
states: np.ndarray = data['states']
actions: np.ndarray = data['actions']
saliency: np.ndarray = data['saliency']

num_iter = states.shape[0]

# standard plot
for j in range(8):
    plt.plot(saliency[:, j], label=state_labels[j])
plt.legend()
plt.savefig('output/saliency.png')

# stacked image
plt.clf()
plt.stackplot(np.arange(num_iter), saliency.T, labels=state_labels)
plt.legend()
plt.savefig('output/saliency_stacked.png')

plt.clf()
# canvas size
x_min, x_max = np.min(states[:, 0]), np.max(states[:, 0])
y_min, y_max = np.min(states[:, 1]), np.max(states[:, 1])

x_box_min = x_min - 0.1 * (x_max - x_min)
x_box_max = x_max + 0.1 * (x_max - x_min)
x_box_max = max(x_box_max, 0.25)
y_box_max = y_max + 0.1 * (y_max - y_min)

arrow_len = 0.5 * min(x_max - x_min, y_max - y_min)

# animation writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

# animation
fig, (ax1, ax2) = plt.subplots(figsize=(8, 4), ncols=2, dpi=300)
ax1: plt.Axes = ax1
ax2: plt.Axes = ax2

ax1.set_aspect('equal')
ax1.set_xlim(x_box_min, x_box_max)
ax1.set_ylim(0, y_box_max)

ax2.set_xticks(range(0, 8))
ax2.set_xticklabels(state_labels, rotation=-45)
ax2.set_ylim(0, 1.1)

bar_plot = ax2.bar(range(8), saliency[100])


def animate(t):
    # animate arrow
    if ax1.patches:
        ax1.patches.pop(0)
    patch = plt.Arrow(states[t, 0], states[t, 1],
                      arrow_len * np.sin(states[t, 4]),
                      arrow_len * np.cos(states[t, 4]),
                      width=0.1)
    ax1.add_patch(patch)
    # animate bar graph
    for j, b in enumerate(bar_plot):
        b.set_height(saliency[t, j])
    return patch,


anim = animation.FuncAnimation(fig, animate, frames=num_iter)
anim.save('output/saliency.mp4', writer=writer)
