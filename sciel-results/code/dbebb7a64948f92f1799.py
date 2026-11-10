"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score_centres(c):
    """Compute the achievable radius for a set of centres in [-0.5, 0.5]^3."""
    # Wall clearance
    wall = 0.5 - np.max(np.abs(c), axis=1)
    if np.any(wall < 0):
        return 0.0
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(min_pair / 2.0, np.min(wall))


def _relax(c, max_iter=2000, dt_init=0.005, dt_min=1e-6, time_budget=None):
    """
    Force-directed relaxation: spheres repel each other (inverse-square) and
    walls repel centres that are too close.  We iterate until forces are small
    or time budget is exhausted.
    """
    c = c.copy()
    dt = dt_init
    best_c = c.copy()
    best_r = _score_centres(best_c)
    t0 = time.time()

    for it in range(max_iter):
        if time_budget is not None and (time.time() - t0) > time_budget:
            break

        # Compute forces
        forces = np.zeros_like(c)

        # Pairwise repulsion (soft, inverse-square with a floor to avoid blow-up)
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, 1e12)  # avoid division by zero
        # Force magnitude ~ 1/dist^2, direction along diff
        # Use a softening to keep forces bounded
        soft = 0.02
        w = 1.0 / (dist ** 2 + soft ** 2)
        np.fill_diagonal(w, 0.0)
        forces -= np.sum(w[:, :, None] * diff, axis=1) * 0.5

        # Wall repulsion: push centres away from walls
        for axis in range(3):
            # Left wall at -0.5
            d_left = c[:, axis] + 0.5
            mask = d_left < 0.1
            forces[mask, axis] += 50.0 * (0.1 - d_left[mask]) ** 2
            # Right wall at +0.5
            d_right = 0.5 - c[:, axis]
            mask = d_right < 0.1
            forces[mask, axis] -= 50.0 * (0.1 - d_right[mask]) ** 2

        # Gradient step
        c += dt * forces

        # Clamp to cube (hard constraint)
        c = np.clip(c, -0.499, 0.499)

        # Evaluate
        r = _score_centres(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

        # Decay step size
        dt *= 0.998
        if dt < dt_min:
            break

    return best_c, best_r


def construct():
    """
    Build 100 equal-sphere packing in the unit cube [-0.5, 0.5]^3.
    Strategy:
      1. Start from a perturbed cubic grid (5x5x5 = 125 sites, take 100).
      2. Run force-directed relaxation to escape the grid's local minimum.
      3. Multiple random restarts with different perturbations; keep the best.
    """
    t_start = time.time()
    time_limit = 110.0  # safety margin before 120s kill

    best_c = None
    best_r = 0.0

    # Generate several random restarts
    rng = np.random.RandomState(42)

    for restart in range(30):
        if time.time() - t_start > time_limit - 5:
            break

        # Build a 5x5x5 grid and randomly pick 100 of the 125 sites
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])

        # Randomly select 100 sites
        idx = rng.choice(len(grid), size=N, replace=False)
        c = grid[idx].copy()

        # Add small random perturbation to break symmetry
        c += rng.randn(N, 3) * 0.005
        c = np.clip(c, -0.49, 0.49)

        # Relax
        time_left = time_limit - (time.time() - t_start) - 2.0
        if time_left < 1.0:
            break
        c_relaxed, r = _relax(c, max_iter=1500, time_budget=time_left)

        if r > best_r:
            best_r = r
            best_c = c_relaxed

    if best_c is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]

    return best_c
# EVOLVE-BLOCK-END
