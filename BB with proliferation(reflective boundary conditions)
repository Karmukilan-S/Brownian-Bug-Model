import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# PARAMETERS
# ----------------------------
L = 1
N0 = 20000
cycles = 100
p = 0.4
q = 0.4
Delta = 0.001
U = 0
k = 2 * np.pi / L

# ----------------------------
# INITIAL CONDITION
# ----------------------------
x = np.random.rand(N0) * L
y = np.random.rand(N0) * L
colors = y.copy()

x_initial = x.copy()
y_initial = y.copy()
colors_initial = colors.copy()

# ----------------------------
# ROBUST REFLECTION FUNCTION
# ----------------------------
def reflect(arr, L):
    arr = np.mod(arr, 2*L)
    arr = np.where(arr > L, 2*L - arr, arr)
    return arr

# ----------------------------
# TIME EVOLUTION
# ----------------------------
for t in range(cycles):

    N = len(x)
    if N == 0:
        break

    # ---- (1) Birth / Death ----
    survivors_x = []
    survivors_y = []
    survivors_c = []

    for i in range(N):

        r = np.random.rand()

        if r < q:
            continue

        survivors_x.append(x[i])
        survivors_y.append(y[i])
        survivors_c.append(colors[i])

        if r < q + p:
            survivors_x.append(x[i])
            survivors_y.append(y[i])
            survivors_c.append(colors[i])

    x = np.array(survivors_x)
    y = np.array(survivors_y)
    colors = np.array(survivors_c)

    if len(x) == 0:
        break

    # ---- (2) Brownian motion ----
    x += np.random.randn(len(x)) * Delta
    y += np.random.randn(len(y)) * Delta

    x = reflect(x, L)
    y = reflect(y, L)

    # ---- (3) Stirring ----
    J = np.random.rand() * 2*np.pi
    v = np.random.rand() * 2*np.pi

    x_new = x + (U/2.0) * np.cos(k*y + J)
    y_new = y + (U/2.0) * np.cos(k*x_new + v)

    x = reflect(x_new, L)
    y = reflect(y_new, L)

# ----------------------------
# PLOT
# ----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

ax[0].scatter(x_initial, y_initial, c=colors_initial,
              s=0.5, cmap='viridis', edgecolors='none')
ax[0].set_title("Initial Configuration")
ax[0].set_xlim(0, L)
ax[0].set_ylim(0, L)
ax[0].set_aspect('equal')

if len(x) > 0:
    ax[1].scatter(x, y, c=colors,
                  s=0.5, cmap='viridis', edgecolors='none')

ax[1].set_title("Final Configuration (Reflecting BC)")
ax[1].set_xlim(0, L)
ax[1].set_ylim(0, L)
ax[1].set_aspect('equal')

plt.tight_layout()
plt.show()
