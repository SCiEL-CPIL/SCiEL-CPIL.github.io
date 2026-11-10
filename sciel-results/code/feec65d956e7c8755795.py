"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    t_start = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n = 8

    # Vectorized log-objective: maximize sum of log(distances) over ordered pairs
    # (equivalent to maximizing product of distances; scaling handled at end)
    def log_obj(pts):
        # pts shape (8, 2)
        # Compute all pairwise distances (ordered, including i!=j)
        # Use broadcasting: diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (8,8,2)
        d = np.sqrt(np.sum(diff * diff, axis=2))  # (8,8)
        # Zero out diagonal (distance 0 -> log 0 = -inf)
        np.fill_diagonal(d, 1.0)  # avoid log(0)
        # Sum log of all off-diagonal distances
        return np.sum(np.log(d))

    # We want to MAXIMIZE log_obj, so minimize -log_obj
    def neg_log_obj(pts):
        return -log_obj(pts)

    # Bounds: coordinates in [-10, 10] (well within 1e6 limit, and diameter scaling
    # makes the absolute scale irrelevant; the score is scale-invariant after
    # normalizing to diameter 2)
    bounds = [(-10.0, 10.0)] * (2 * n)

    # Run differential evolution with global search
    # Use a reasonable number of iterations and population size
    # DE is stochastic, so set seed for reproducibility within the run
    best_result = None
    best_score = -np.inf

    # Try a few different seeds for DE to increase diversity
    for seed in [42, 123, 456]:
        if time.time() > deadline - 30.0:
            break
        try:
            res = differential_evolution(
                neg_log_obj,
                bounds,
                seed=seed,
                maxiter=200,
                popsize=20,
                tol=1e-12,
                polish=True,  # use L-BFGS-B polish at the end
                init='sobol',  # better space-filling initialization
                updating='deferred',
                workers=1,
            )
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            continue

    if best_result is None:
        # Fallback: regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    pts = best_result.reshape(n, 2)

    # Verify distinctness and finite coordinates
    # Check all pairwise distances > 0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, 1.0)
    if np.min(d) < 1e-10 or not np.all(np.isfinite(pts)):
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    # Scale to diameter 2
    # Diameter is the maximum pairwise distance
    diam = np.max(d)
    if diam < 1e-10:
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
    pts_scaled = pts * (2.0 / diam)

    # Verify coordinates are within bounds
    if np.max(np.abs(pts_scaled)) >= 1e6:
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    return pts_scaled.tolist()
# EVOLVE-BLOCK-END
