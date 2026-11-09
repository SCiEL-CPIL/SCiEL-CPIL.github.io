"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE_SECONDS = 115.0  # safety margin before 120s kill


def _objective(params, n):
    """
    Objective for L-BFGS-B: we want to maximize sum of radii.
    L-BFGS-B minimizes, so we return negative sum of radii.
    Constraints are handled via bounds and penalty if needed, but L-BFGS-B
    only supports bounds. We'll use a soft penalty for overlaps and boundary violations.
    """
    x = params[:n]
    y = params[n:2 * n]
    r = params[2 * n:3 * n]

    # Penalty for boundary violations
    pen = 0.0
    # x in [r, 1-r], y in [r, 1-r]
    # If x < r, penalty (r - x)^2; if x > 1-r, penalty (x - (1-r))^2
    # Vectorized
    lo_x = r
    hi_x = 1.0 - r
    # penalty for x < lo_x
    viol_x_lo = np.maximum(0.0, lo_x - x)
    viol_x_hi = np.maximum(0.0, x - hi_x)
    pen += np.sum(viol_x_lo ** 2) + np.sum(viol_x_hi ** 2)

    lo_y = r
    hi_y = 1.0 - r
    viol_y_lo = np.maximum(0.0, lo_y - y)
    viol_y_hi = np.maximum(0.0, y - hi_y)
    pen += np.sum(viol_y_lo ** 2) + np.sum(viol_y_hi ** 2)

    # Penalty for overlaps
    # For each pair i,j: distance < r_i + r_j
    # Vectorized pairwise distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    min_dist = r[:, None] + r[None, :]
    # Only consider i < j to avoid double counting and self
    iu = np.triu_indices(n, k=1)
    d_ij = dist[iu]
    r_ij = min_dist[iu]
    overlap = np.maximum(0.0, r_ij - d_ij)
    pen += np.sum(overlap ** 2)

    # Objective: -sum(r) + penalty * weight
    # We want penalty to dominate when constraints are violated
    weight = 100.0
    return -np.sum(r) + weight * pen


def _evaluate(params, n):
    """Compute the actual objective value (sum of radii) and validity."""
    x = params[:n]
    y = params[n:2 * n]
    r = params[2 * n:3 * n]

    # Check boundaries
    if np.any(x < r) or np.any(x > 1.0 - r) or np.any(y < r) or np.any(y > 1.0 - r):
        return 0.0, False

    # Check overlaps
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    min_dist = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < min_dist[iu] - 1e-9):
        return 0.0, False

    return np.sum(r), True


def construct():
    n = N
    start_time = time.time()

    # Initial guess: grid-based
    side = int(np.ceil(np.sqrt(n)))
    x0 = np.zeros(3 * n)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            x0[idx] = (i + 0.5) / side
            x0[n + idx] = (j + 0.5) / side
            x0[2 * n + idx] = 0.5 / side - 0.01
            idx += 1

    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n

    best_params = x0.copy()
    best_val, best_valid = _evaluate(best_params, n)

    # Multiple restarts with different initial guesses
    rng = np.random.default_rng(42)

    def try_optimize(x_init, maxiter=2000, tol=1e-8):
        nonlocal best_params, best_val, best_valid
        try:
            res = minimize(_objective, x_init, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000})
            if res.success or res.fun < _objective(best_params, n):
                val, valid = _evaluate(res.x, n)
                if valid and val > best_val:
                    best_params = res.x.copy()
                    best_val = val
                    best_valid = True
                elif not best_valid and valid:
                    best_params = res.x.copy()
                    best_val = val
                    best_valid = True
        except Exception:
            pass

    # First, try the grid initial
    try_optimize(x0)

    # Additional restarts with random perturbations
    num_restarts = 20
    for k in range(num_restarts):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break

        # Random initial guess
        x_rand = np.zeros(3 * n)
        # Place circles in random positions with small radii
        for i in range(n):
            x_rand[i] = 0.1 + 0.8 * rng.random()
            x_rand[n + i] = 0.1 + 0.8 * rng.random()
            x_rand[2 * n + i] = 0.05 + 0.1 * rng.random()

        try_optimize(x_rand, maxiter=1000)

    # Polish step: high-precision optimization on the best candidate
    if time.time() - start_time < DEADLINE_SECONDS - 2:
        if best_valid:
            try:
                res = minimize(_objective, best_params, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 50000})
                val, valid = _evaluate(res.x, n)
                if valid and val > best_val:
                    best_params = res.x.copy()
                    best_val = val
                    best_valid = True
            except Exception:
                pass

    # Ensure we return a valid configuration
    if not best_valid:
        # Fallback to a simple valid configuration
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= n:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.05])
                idx += 1
        return np.array(cells)

    result = np.zeros((n, 3))
    result[:, 0] = best_params[:n]
    result[:, 1] = best_params[n:2 * n]
    result[:, 2] = best_params[2 * n:3 * n]

    # Final validation
    val, valid = _evaluate(result.ravel(), n)
    if not valid:
        # Fallback
        side = int(np.ceil(np.sqrt(n)))
        cells = []
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= n:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.05])
                idx += 1
        return np.array(cells)

    return result
# EVOLVE-BLOCK-END
