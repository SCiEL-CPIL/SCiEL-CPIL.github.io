"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Minimum distance to boundary
    min_boundary = 0.5 - np.abs(centers).max()
    if min_boundary <= 0:
        return 0.0
    # Minimum pairwise distance / 2
    if n < 2:
        return min_boundary
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    return min(min_boundary, min_pair)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin

    # Start with a grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()

    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)

    # Simulated annealing parameters
    T0 = 0.05
    T_end = 0.001
    cooling_rate = 0.999
    T = T0

    np.random.seed(42)

    while time.time() < deadline and T > T_end:
        # Generate a random perturbation
        move_type = np.random.random()
        if move_type < 0.7:
            # Small random displacement
            idx = np.random.randint(N)
            delta = np.random.normal(0, T, 3)
            new_centers = centers.copy()
            new_centers[idx] += delta
            # Clamp to cube
            new_centers = np.clip(new_centers, -0.5, 0.5)
        elif move_type < 0.9:
            # Swap two centers
            i, j = np.random.choice(N, 2, replace=False)
            new_centers = centers.copy()
            new_centers[[i, j]] = new_centers[[j, i]]
        else:
            # Larger random jump
            idx = np.random.randint(N)
            delta = np.random.normal(0, T * 2, 3)
            new_centers = centers.copy()
            new_centers[idx] += delta
            new_centers = np.clip(new_centers, -0.5, 0.5)

        new_radius = _compute_radius(new_centers)
        delta_r = new_radius - best_radius

        # Accept if better, or with probability exp(delta_r / T) if worse
        if delta_r >= 0 or np.random.random() < np.exp(delta_r / T):
            centers = new_centers
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()

        T *= cooling_rate

    # Final polish: try small local moves around best solution
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        polish_centers = best_centers.copy()
        polish_radius = best_radius
        T_polish = 0.005
        n_polish = int(min(5000, remaining_time * 100))
        for _ in range(n_polish):
            if time.time() > deadline - 2:
                break
            idx = np.random.randint(N)
            delta = np.random.normal(0, T_polish, 3)
            new_centers = polish_centers.copy()
            new_centers[idx] += delta
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_radius = _compute_radius(new_centers)
            if new_radius > polish_radius:
                polish_centers = new_centers
                polish_radius = new_radius
                T_polish *= 0.99
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers.copy()

    return best_centers
# EVOLVE-BLOCK-END
