"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius_for_centers(centers):
    """Compute the radius: min of half pairwise distances and wall distances."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distances: min over each coordinate of (0.5 - |c|)
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_wall, min_pair)


def _neg_radius(centers_flat):
    """Negative radius for minimization."""
    centers = centers_flat.reshape(N, 3)
    return -_radius_for_centers(centers)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Initial configuration: use a jittered grid or random points
    # Let's try a few different initializations and pick the best

    def _make_initial(seed):
        rng = np.random.RandomState(seed)
        # Start with a grid and add jitter
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
        # Add small random jitter
        centers += rng.uniform(-0.01, 0.01, size=centers.shape)
        # Clip to stay inside cube
        centers = np.clip(centers, -0.49, 0.49)
        return centers

    def _make_random_initial(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.49, 0.49, size=(N, 3))

    best_centers = None
    best_radius = 0.0

    # Try multiple initializations
    initializations = [
        _make_initial(42),
        _make_random_initial(1),
        _make_random_initial(2),
        _make_random_initial(3),
    ]

    for init_centers in initializations:
        if time.time() > deadline - 10.0:
            break

        current = init_centers.copy()
        current_radius = _radius_for_centers(current)

        # Local optimization using Nelder-Mead on flattened centers
        x0 = current.flatten()
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            try:
                result = minimize(
                    _neg_radius,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'fatol': 1e-10, 'xatol': 1e-10},
                    timeout=20.0
                )
                if result.fun < -current_radius + 1e-12:
                    current = result.x.reshape(N, 3)
                    current_radius = -result.fun
            except Exception:
                pass
            # If no improvement, break
            if current_radius <= best_radius + 1e-12:
                break

        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = current.copy()

    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.499, 0.499)
    return best_centers
# EVOLVE-BLOCK-END
