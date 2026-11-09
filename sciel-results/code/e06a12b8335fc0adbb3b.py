"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill

    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].copy()

    def radius(c):
        """Compute the minimum radius for configuration c."""
        c = c.reshape(N, 3)
        # Wall distances
        wall_dists = np.min(0.5 - np.abs(c), axis=1)
        # Pairwise distances
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        return min(np.min(wall_dists), min_pair_dist / 2.0)

    def neg_radius(c):
        return -radius(c)

    best_x = x0.copy()
    best_r = radius(x0)

    # Local optimization with multiple restarts
    n_restarts = 0
    max_restarts = 10

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the best solution found so far
            x_start = best_x + np.random.randn(N, 3) * 0.02

        # Clip to stay within bounds
        x_start = np.clip(x_start, -0.49, 0.49)

        try:
            result = minimize(
                neg_radius,
                x_start,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'maxls': 50}
            )

            r_result = -result.fun
            if r_result > best_r + 1e-12:
                best_r = r_result
                best_x = result.x
                n_restarts = 0  # reset counter on improvement
            else:
                n_restarts += 1

        except Exception:
            n_restarts += 1

        # Check deadline
        if time.time() > deadline:
            break

    # Final validation and clipping
    best_x = np.clip(best_x, -0.5, 0.5)

    return best_x
# EVOLVE-BLOCK-END
