import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree  # for neighbor queries


def calculate_gr(positions, L, dr=0.001, r_max=None):
    """
    Computes the pair correlation function g(r) for a 2D system.

    Parameters:
    - positions: (M, 2) array of bug positions
    - L: Box size (for periodic boundaries)
    - dr: Bin width for the histogram
    - r_max: Maximum distance to consider (usually L/2)
    """
    M = len(positions)
    if M < 2:
        return np.zeros(1), np.zeros(1)

    if r_max is None:
        r_max = L / 2.0  # Common convention for PBC

    # 1. Use cKDTree to find all pairs within r_max under Periodic Boundary Conditions
    tree = cKDTree(positions, boxsize=L)

    # count_neighbors(tree, r) returns the number of pairs within distance r.
    # We want a histogram, so we calculate counts for all bins.
    r_bins = np.arange(0, r_max + dr, dr)
    # This returns the cumulative number of pairs
    cumulative_pairs = tree.count_neighbors(tree, r_bins)

    # 2. Get counts in each bin (non-cumulative)
    # We subtract 1 because count_neighbors includes the self-pair (distance=0)
    pair_counts = np.diff(cumulative_pairs)

    # 3. Normalize
    # Ideal gas density (mean density)
    rho = M / (L**2)

    # Midpoints of bins for plotting
    r_centers = r_bins[:-1] + dr/2

    # Area of each circular shell: pi * (r_out^2 - r_in^2)
    shell_areas = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)

    # g(r) = (Observed Pairs in shell) / (Expected Pairs in shell for uniform density)
    # Expected pairs = rho * shell_area * number_of_particles
    # Note: count_neighbors counts each pair twice (i,j and j,i),
    # but also includes the self-count in its cumulative logic.
    # The diff correctly gives the number of pairs including double counting.
    gr = pair_counts / (shell_areas * rho * M)

    return r_centers, gr

def simulate_ABB(num_steps, dt, L,
                 M0, v0, D_r, D_theta,
                 p0, q, R, N_s,
                 variant, seed=None,
                 tree_update_freq=10, max_capacity=40000):
    """
    Simulate the Active Brownian Bug model (Optimized Version).
    Added 'tree_update_freq' to reduce KDTree overhead.
    Added 'max_capacity' to pre-allocate memory and prevent vstack bottlenecks.
    """
    if seed is not None:
        np.random.seed(seed)

    # --- OPTIMIZATION 1: MEMORY PREALLOCATION ---
    # We allocate massive arrays once, rather than using np.vstack every loop
    positions = np.zeros((max_capacity, 2))
    theta = np.zeros(max_capacity)
    p_i = np.zeros(max_capacity) # Track reproduction rates so newborns can inherit them

    # Initialize M0 agents
    positions[:M0] = np.random.rand(M0, 2) * L
    theta[:M0] = np.random.rand(M0) * 2 * np.pi
    current_N = M0

    # Precompute noise scales
    sqrt2Dr = np.sqrt(2 * D_r * dt)
    sqrt2Dtheta = np.sqrt(2 * D_theta * dt)

    # Time series arrays
    N_ts = np.zeros(num_steps, dtype=int)
    S_ts = np.zeros(num_steps, dtype=float)

    for t in range(num_steps):
        # Safety catch for extinction
        if current_N == 0:
            N_ts[t:] = 0
            S_ts[t:] = 0
            break

        # Create a "view" of only the currently alive agents.
        # This is instantaneous and doesn't copy data in memory.
        pos_active = positions[:current_N]
        theta_active = theta[:current_N]

        # (1) Move agents
        dx = v0 * np.cos(theta_active) * dt + np.random.randn(current_N) * sqrt2Dr
        dy = v0 * np.sin(theta_active) * dt + np.random.randn(current_N) * sqrt2Dr
        pos_active[:, 0] = (pos_active[:, 0] + dx) % L
        pos_active[:, 1] = (pos_active[:, 1] + dy) % L
        theta_active[:] = (theta_active + np.random.randn(current_N) * sqrt2Dtheta) % (2 * np.pi)

        # (2) Compute reproduction rates (OPTIMIZATION 2: LAZY EVALUATION)
        if variant == 'ND':
            # Only rebuild the expensive KDTree every N steps
            if t % tree_update_freq == 0:
                tree = cKDTree(pos_active, boxsize=L)
                Ni = np.array(tree.query_ball_point(pos_active, r=R, return_length=True)) - 1
                p_i[:current_N] = np.maximum(p0 * (1 - Ni / N_s), 0.0)
        else:
            p_i[:current_N] = p0

        # (3) Birth and death events
        r = np.random.rand(current_N)
        current_p_i = p_i[:current_N]

        birth_mask = (r < current_p_i * dt)
        death_mask = ((r >= current_p_i * dt) & (r < current_p_i * dt + q * dt))
        survivors_mask = ~death_mask

        # Get data for survivors
        surv_pos = pos_active[survivors_mask]
        surv_theta = theta_active[survivors_mask]
        surv_pi = current_p_i[survivors_mask]

        # Get data for newborns (they duplicate the parent's position, theta, and p_i)
        newborns_pos = pos_active[birth_mask]
        newborns_theta = theta_active[birth_mask]
        newborns_pi = current_p_i[birth_mask]

        new_N = len(surv_pos) + len(newborns_pos)

        # Safety catch to prevent array index errors
        if new_N > max_capacity:
            raise ValueError(f"Population {new_N} exceeded max_capacity! Increase max_capacity parameter.")

        # Update the master pre-allocated arrays in-place
        positions[:len(surv_pos)] = surv_pos
        positions[len(surv_pos):new_N] = newborns_pos

        theta[:len(surv_theta)] = surv_theta
        theta[len(surv_theta):new_N] = newborns_theta

        p_i[:len(surv_pi)] = surv_pi
        p_i[len(surv_pi):new_N] = newborns_pi

        current_N = new_N

        # Record metrics
        N_ts[t] = current_N
        S_ts[t] = np.abs(np.mean(np.exp(1j * theta[:current_N])))

    # Return only the sliced data of agents that are actually alive at the end
    return positions[:current_N], theta[:current_N], N_ts, S_ts
# Example usage:
L = 1.0          # box size
dt = 0.01
steps = 1000000  # total steps
for (v_0,D_r) in [(0.356e-3,0.356e-5),(3.557e-3,0.356e-5),(11.25e-3,2e-5),(0.356e-3,6.325e-5)]:

# Set simulation parameters
    params = {
        'M0': 3500,
        'v0': v_0,
        'D_r': D_r,
        'D_theta': 2e-3,
        'p0': 0.85, 'q': 0.15, 'R': 0.1, 'N_s': 50
    }
    # (A) Density-dependent (ND) variant
    pos_nd, th_nd, N_nd_ts, S_nd_ts = simulate_ABB(
        num_steps=steps, dt=dt, L=L, variant='ND', **params, seed=42)

    # (B) Constant-birth variant
    # Use same p0,q or adjust to avoid runaway (e.g. p0=0.15,q=0.15 for similar average N)
    params_const = params.copy()
    params_const['p0'] = 0.15
    params_const['q'] = 0.15
    pos_const, th_const, N_const_ts, S_const_ts = simulate_ABB(
        num_steps=steps, dt=dt, L=L, variant='const', **params_const, seed=42)
    plt.figure(figsize=(3,2))
    r_const,g_const=calculate_gr(pos_const,L,0.01,L/2)
    r_nd,g_nd=calculate_gr(pos_nd,L,0.01,L/2)

    plt.plot(r_nd,g_nd,label="density dependent birth")
    plt.plot(r_const,g_const,label="Constant birth")
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Pair correlation function")
    plt.xlim(0,L/2)
    plt.ylim(0,3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(3,2))
    time=np.arange(steps)*dt
    plt.plot(time, N_nd_ts, label="Density-dependent birth")
    plt.plot(time, N_const_ts, label="Constant birth")

    plt.xlabel("Time")
    plt.ylabel("Population N(t)")
    plt.title("Population dynamics in ABB model")
    plt.legend()

    plt.show()

    plt.figure(figsize=(3,2))

    plt.plot(time, S_nd_ts, label="Density-dependent birth")
    plt.plot(time, S_const_ts, label="Constant birth")

    plt.xlabel("Time")
    plt.ylabel("Polar order S(t)")
    plt.title("Polar order parameter")
    plt.legend()

    plt.show()

    fig, ax = plt.subplots(1,2, figsize=(10,4))

    ax[0].scatter(pos_nd[:,0], pos_nd[:,1], s=3)
    ax[0].set_title("Density-dependent")

    ax[1].scatter(pos_const[:,0], pos_const[:,1], s=3)
    ax[1].set_title("Constant birth")

    for a in ax:
        a.set_xlim(0,L)
        a.set_ylim(0,L)

    plt.show()
