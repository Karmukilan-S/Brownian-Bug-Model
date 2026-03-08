import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# PARAMETERS
# ----------------------------
L = 1            # domain size
N0 = 20000              # initial population
cycles = 1000   # number of cycles
p = 0.1              # birth probability per cycle
q = 0.1             # death probability per cycle
Delta = 0.001          # Brownian RMS displacement
U = 0.05           # stirring amplitude
tau = 1.0             # cycle duration
k = 2*np.pi / L       # wave number

# ----------------------------
# INITIAL CONDITION
# ----------------------------
positions = np.random.rand(N0, 2) * L

# Color varies smoothly with y
colors = positions[:,1] / L
initial_positions = positions.copy()
initial_colors = colors.copy()

# ----------------------------
# MAIN LOOP
# ----------------------------
for c in range(cycles):

    if len(positions) == 0:
        break

    N = len(positions)

    # ----------------------------
    # (1) Birth–Death
    # ----------------------------
    survivors = []
    survivor_colors = []
    births = []
    birth_colors = []

    for i in range(N):
        r = np.random.rand()

        if r < q:
            continue  # death

        survivors.append(positions[i])
        survivor_colors.append(colors[i])

        if r < q + p:
            births.append(positions[i].copy())
            birth_colors.append(colors[i])

    if len(survivors) + len(births) == 0:
        positions = np.empty((0,2))
        colors = np.array([])
        break

    positions = np.array(survivors + births)
    colors = np.array(survivor_colors + birth_colors)

    # ----------------------------
    # (2) Brownian Motion
    # ----------------------------
    positions += Delta * np.random.randn(len(positions), 2)

    # periodic wrap
    positions %= L

    # ----------------------------
    # (3) Advective Stirring
    # ----------------------------
    J = np.random.uniform(0, 2*np.pi)
    v = np.random.uniform(0, 2*np.pi)

    x = positions[:,0].copy()
    y = positions[:,1].copy()

    x_new = x + (U * tau / 2.0) * np.cos(k*y + J)
    y_new = y + (U * tau / 2.0) * np.cos(k*x_new + v)

    positions[:,0] = x_new % L
    positions[:,1] = y_new % L

# ----------------------------
# PLOT
# ----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12,5))

ax[0].scatter(initial_positions[:,0],
              initial_positions[:,1],
              c=initial_colors,
              s=0.5,
              cmap='viridis')
ax[0].set_xlim(0,L)
ax[0].set_ylim(0,L)
ax[0].set_title("Initial")

if len(positions) > 0:
    ax[1].scatter(positions[:,0],
                  positions[:,1],
                  c=colors,
                  s=0.5,
                  cmap='viridis')

ax[1].set_xlim(0,L)
ax[1].set_ylim(0,L)
ax[1].set_title("Final")

plt.tight_layout()
plt.show()
