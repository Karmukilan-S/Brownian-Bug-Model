import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# PARAMETERS
# ----------------------------
L = 1                 # domain size
N0 = 20000           # initial population
cycles = 100         # number of cycles
p = 0.45              # birth probability
q = 0.45               # death probability

#Delta = 0.001          # Brownian RMS displacement

# Active motion parameters
#v0 = 0.0001*0.356             # self propulsion speed
D_0 = 0.002              # rotational diffusion

# Stirring parameters
U = 0.00
tau = 1
k = 2*np.pi / L

# ----------------------------
# INITIAL CONDITION
# ----------------------------
positions = np.random.rand(N0, 2) * L
theta = np.random.rand(N0) * 2*np.pi   # orientation of each particle

# Color varies smoothly with y
colors = positions[:,1] / L

initial_positions = positions.copy()
initial_colors = colors.copy()


S_series = []
time_series = []
# ----------------------------
# MAIN LOOP
# ----------------------------
for c in range(cycles):

    if len(positions) == 0:
        break

    N = len(positions)

    M = len(theta)

    S_t = np.abs(np.sum(np.exp(1j * theta))) / M

    S_series.append(S_t)
    time_series.append(c)

    # ----------------------------
    # (1) Birth–Death
    # ----------------------------
    survivors = []
    survivor_colors = []
    survivor_theta = []

    births = []
    birth_colors = []
    birth_theta = []

    for i in range(N):

        r = np.random.rand()

        if r < q:
            continue

        survivors.append(positions[i])
        survivor_colors.append(colors[i])
        survivor_theta.append(theta[i])

        if r < q + p:
            births.append(positions[i].copy())
            birth_colors.append(colors[i])
            birth_theta.append(theta[i])

    if len(survivors) + len(births) == 0:
        positions = np.empty((0,2))
        colors = np.array([])
        theta = np.array([])
        break

    positions = np.array(survivors + births)
    colors = np.array(survivor_colors + birth_colors)
    theta = np.array(survivor_theta + birth_theta)

    # ----------------------------
    # (2) Brownian Motion
    # ----------------------------
    positions += Delta * np.random.randn(len(positions), 2)

    # ----------------------------
    # (3) Self Propulsion
    # ----------------------------
    theta += np.sqrt(2*D_0) * np.random.randn(len(theta))

    positions[:,0] += v0 * np.cos(theta)
    positions[:,1] += v0 * np.sin(theta)
    
   
    # periodic wrap
    positions %= L

    # ----------------------------
    # (4) Advective Stirring
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
              s=1,
              cmap='viridis')

ax[0].set_xlim(0,L)
ax[0].set_ylim(0,L)
ax[0].set_title("Initial")


if len(positions) > 0:

    ax[1].scatter(positions[:,0],
                  positions[:,1],
                  c=colors,
                  s=1,
                  cmap='viridis')

    # ----------------------------
    # Orientation lines for ALL particles
    # ----------------------------

    line_len = 0.005 * L   # very small line length

    x = positions[:,0]
    y = positions[:,1]

    x2 = x + line_len * np.cos(theta)
    y2 = y + line_len * np.sin(theta)

    for i in range(len(positions)):
        ax[1].plot([x[i], x2[i]],
                   [y[i], y2[i]],
                   color=plt.cm.viridis(colors[i]),
                   linewidth=0.3)

ax[1].set_xlim(0,L)
ax[1].set_ylim(0,L)
ax[1].set_title("Final")

plt.tight_layout()
plt.show()

# ==========================================================
# CORRELATION ANALYSIS
# ==========================================================

if len(positions) > 10:

    N = len(positions)
    rho = N / (L*L)

    # --------------------------------------------------
    # Pair Correlation Function g(r)
    # --------------------------------------------------

    bins = 150
    r_max = L/2
    dr = r_max / bins

    g = np.zeros(bins)

    for i in range(N):

        dx = positions[i,0] - positions[:,0]
        dy = positions[i,1] - positions[:,1]

        # periodic boundary correction
        dx -= L*np.round(dx/L)
        dy -= L*np.round(dy/L)

        r = np.sqrt(dx**2 + dy**2)

        hist, edges = np.histogram(r, bins=bins, range=(0,r_max))
        g += hist

    r_centers = (edges[:-1] + edges[1:]) / 2

    shell_area = 2*np.pi*r_centers*dr
    ideal = rho * shell_area * N

    g = g / ideal

   

    # --------------------------------------------------
    # Plot g(r)
    # --------------------------------------------------

    plt.figure(figsize=(6,5))
    plt.plot(r_centers, g)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Pair Correlation Function")
    plt.grid()
    plt.show()

    # --------------------------------------------------
    # Plot S(k)
    # --------------------------------------------------



plt.figure(figsize=(6,4))
plt.plot(time_series, S_series)
plt.xlabel("time")
plt.ylabel("S(t)")
plt.title("Polarization Order Parameter")
plt.show()
