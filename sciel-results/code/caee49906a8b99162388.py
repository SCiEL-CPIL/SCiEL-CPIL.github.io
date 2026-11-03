"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

DEADLINE = 135.0  # seconds; safety margin under the 150 s kill


def _radius(c):
    """Compute the common radius for centres c (N,2) in [-1/2,1/2]^2."""
    n = c.shape[0]
    if n < 2:
        return np.inf
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    # Distance to boundary
    coords = np.abs(c)
    min_boundary = (0.5 - coords).min()
    return min(min_pair, min_boundary)


def _soft_score(c, beta=50.0):
    """A differentiable proxy: large when circles are well-separated and inside."""
    n = c.shape[0]
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
    np.fill_diagonal(dist, np.inf)
    # Soft min of pair distances
    min_pair = -np.logaddexp.reduce(np.log(dist / 2.0 - 1e-12) * 0 + np.log(1.0) * 0)  # placeholder
    # Use a simpler approach: negative log-sum-exp of (1/d)
    inv_d = 1.0 / (dist + 1e-12)
    soft_min_pair = 1.0 / (np.sum(np.exp(beta * inv_d)) ** (1.0 / beta) * beta ** (1.0 / beta))
    # Boundary
    coords = np.abs(c)
    dist_b = 0.5 - coords
    inv_b = 1.0 / (np.abs(dist_b) + 1e-12)
    soft_min_b = 1.0 / (np.sum(np.exp(beta * inv_b)) ** (1.0 / beta) * beta ** (1.0 / beta))
    return min(soft_min_pair, soft_min_b)


def _grid_init():
    """Square grid initialization."""
    m = int(math.ceil(math.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init(seed=None):
    """Random initialization inside the square with some spacing."""
    rng = np.random.default_rng(seed)
    # Start with a jittered grid for better initial spacing
    m = int(math.ceil(math.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N].copy()
    # Add small random jitter
    grid += rng.uniform(-0.02, 0.02, size=grid.shape)
    # Clip to be inside
    np.clip(grid, -0.48, 0.48, out=grid)
    return grid


def _simulate_annealing(c, beta0=20.0, beta1=200.0, steps=5000, temp0=0.01, temp1=0.0001,
                        seed=None):
    """
    Simulated annealing with random perturbations.
    c: (N,2) array of centres
    """
    rng = np.random.default_rng(seed)
    best_c = c.copy()
    best_r = _radius(c)
    current_r = best_r
    n = c.shape[0]

    for step in range(steps):
        if time.time() > DEADLINE:
            break
        # Annealing schedule
        t = step / max(steps - 1, 1)
        temp = temp0 * (temp1 / temp0) ** t
        beta = beta0 + (beta1 - beta0) * t

        # Pick a random point to move
        idx = rng.integers(0, n)
        old_pos = c[idx].copy()

        # Random perturbation
        angle = rng.uniform(0, 2 * math.pi)
        mag = temp * rng.uniform(0.5, 2.0)
        c[idx, 0] += mag * math.cos(angle)
        c[idx, 1] += mag * math.sin(angle)

        # Keep inside the square
        c[idx] = np.clip(c[idx], -0.49, 0.49)

        new_r = _radius(c)

        if new_r > current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = c.copy()
        else:
            # Accept with probability exp((new_r - current_r) / temp)
            delta = new_r - current_r
            if delta < 0 and rng.random() < math.exp(delta / (temp + 1e-12)):
                current_r = new_r
            else:
                c[idx] = old_pos

    return best_c, best_r


def _local_optimize(c, max_iter=2000, tol=1e-10):
    """
    Gradient-free local optimization using coordinate descent + Nelder-Mead style.
    """
    best_c = c.copy()
    best_r = _radius(c)
    n = c.shape[0]
    rng = np.random.default_rng(42)

    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        improved = False

        # Try moving each point in a small neighborhood
        for idx in range(n):
            if time.time() > DEADLINE:
                break
            old = c[idx].copy()
            # Try several random directions
            for _ in range(8):
                angle = rng.uniform(0, 2 * math.pi)
                mag = 0.005 * (1.0 + rng.random())
                c[idx, 0] = old[0] + mag * math.cos(angle)
                c[idx, 1] = old[1] + mag * math.sin(angle)
                np.clip(c[idx], -0.49, 0.49, out=c[idx])
                r = _radius(c)
                if r > best_r + tol:
                    best_r = r
                    best_c = c.copy()
                    improved = True
                    break
            c[idx] = old

        if not improved:
            break

    return best_c, best_r


def _force_based_optimize(c, max_iter=3000, seed=None):
    """
    Simple force-based relaxation: repulsive forces between circles + boundary constraints.
    """
    rng = np.random.default_rng(seed)
    best_c = c.copy()
    best_r = _radius(c)
    n = c.shape[0]
    current_r = best_r
    lr = 0.01

    for it in range(max_iter):
        if time.time() > DEADLINE:
            break

        # Compute forces
        forces = np.zeros_like(c)
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-12)
        np.fill_diagonal(dist, np.inf)

        # Repulsive force: push apart when too close
        for i in range(n):
            for j in range(i + 1, n):
                d = dist[i, j]
                if d < 0.15:  # Only interact when somewhat close
                    f = 1.0 / (d + 1e-6)
                    forces[i] -= f * (c[i] - c[j]) / (d + 1e-6) * 0.01
                    forces[j] += f * (c[i] - c[j]) / (d + 1e-6) * 0.01

        # Boundary repulsion
        for i in range(n):
            for k in range(2):
                if c[i, k] > 0.45:
                    forces[i, k] -= (c[i, k] - 0.45) * 0.5
                elif c[i, k] < -0.45:
                    forces[i, k] += (-0.45 - c[i, k]) * 0.5

        # Update
        c += lr * forces
        np.clip(c, -0.49, 0.49, out=c)

        # Small random jitter to escape local minima
        if it % 100 == 0:
            c += rng.normal(0, 0.001, size=c.shape)
            np.clip(c, -0.49, 0.49, out=c)

        r = _radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

        # Decay learning rate
        lr *= 0.999

    return best_c, best_r


def construct():
    start_time = time.time()

    # Start with grid initialization
    c = _grid_init()
    best_c = c.copy()
    best_r = _radius(c)

    # Phase 1: Simulated annealing from grid
    c_sa, r_sa = _simulate_annealing(c, seed=0)
    if r_sa > best_r:
        best_r = r_sa
        best_c = c_sa.copy()

    if time.time() > DEADLINE - 20:
        return best_c

    # Phase 2: Force-based relaxation
    c_fb, r_fb = _force_based_optimize(best_c, seed=1)
    if r_fb > best_r:
        best_r = r_fb
        best_c = c_fb.copy()

    if time.time() > DEADLINE - 15:
        return best_c

    # Phase 3: Local optimization
    c_lo, r_lo = _local_optimize(best_c)
    if r_lo > best_r:
        best_r = r_lo
        best_c = c_lo.copy()

    if time.time() > DEADLINE - 10:
        return best_c

    # Phase 4: Try random restarts if time permits
    for seed in range(1, 20):
        if time.time() > DEADLINE - 5:
            break
        c_rand = _random_init(seed=seed)
        c_sa2, r_sa2 = _simulate_annealing(c_rand, steps=3000, seed=seed)
        if r_sa2 > best_r:
            best_r = r_sa2
            best_c = c_sa2.copy()
        c_lo2, r_lo2 = _local_optimize(c_sa2)
        if r_lo2 > best_r:
            best_r = r_lo2
            best_c = c_lo2.copy()

    return best_c
# EVOLVE-BLOCK-END
