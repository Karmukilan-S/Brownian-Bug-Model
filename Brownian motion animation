import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# --- Configuration ---
n_side = 40          # 40x40 = 1,600 bugs (Higher Density)
n_frames = 150
step_size = 0.1      # Speed of movement
lim = 5              # Boundary walls at -5 and 5

# 1. Initialize High-Density Grid
x_coords = np.linspace(-4, 4, n_side)
y_coords = np.linspace(-4, 4, n_side)
X, Y = np.meshgrid(x_coords, y_coords)
pos = np.vstack([X.ravel(), Y.ravel()]).T
n_bugs = pos.shape[0]

# 2. Color code by initial Y-position (Vertical Gradient)
colors = plt.cm.plasma(np.linspace(0, 1, n_bugs))

# 3. Setup Plot
fig, ax = plt.subplots(figsize=(7, 7))
scat = ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=10, edgecolors='none')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
# Draw the boundary walls
ax.vlines([-lim, lim], -lim, lim, colors='red', linestyles='dashed')
ax.hlines([-lim, lim], -lim, lim, colors='red', linestyles='dashed')
ax.set_title(f"Brownian Diffusion: {n_bugs} Bugs with Reflective Walls")
plt.close()

# 4. Update Function with Reflection Logic
def update(frame):
    global pos
    # Standard Brownian step
    noise = np.random.normal(0, step_size, size=pos.shape)
    pos += noise

    # --- Reflective Boundary Logic ---
    # Check X boundaries
    out_of_bounds_x_upper = pos[:, 0] > lim
    out_of_bounds_x_lower = pos[:, 0] < -lim
    pos[out_of_bounds_x_upper, 0] = 2 * lim - pos[out_of_bounds_x_upper, 0]
    pos[out_of_bounds_x_lower, 0] = -2 * lim - pos[out_of_bounds_x_lower, 0]

    # Check Y boundaries
    out_of_bounds_y_upper = pos[:, 1] > lim
    out_of_bounds_y_lower = pos[:, 1] < -lim
    pos[out_of_bounds_y_upper, 1] = 2 * lim - pos[out_of_bounds_y_upper, 1]
    pos[out_of_bounds_y_lower, 1] = -2 * lim - pos[out_of_bounds_y_lower, 1]

    scat.set_offsets(pos)
    return (scat,)

# 5. Render
ani = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=True)
HTML(ani.to_html5_video())
