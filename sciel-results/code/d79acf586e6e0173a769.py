"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_SECONDS = 130.0  # Safety margin before 150s kill


def _compute_objective(params):
    """
    Compute negative sum of radii (to minimize).
    params: array of shape (3N,) -> (x0,y0,r0, x1,y1,r1, ...)
    Returns: (-sum_r, violations)
    """
    n = N
    pts = params.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]

    # Boundary violations: circles must be inside [0,1]^2
    # x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    bnd_viol = np.maximum(0, -x + r) + np.maximum(0, x + r - 1) + \
               np.maximum(0, -y + r) + np.maximum(0, y + r - 1)

    # Pairwise non-overlap: ||pi - pj|| >= ri + rj
    # Compute all pairwise distances
    # Use vectorized computation
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]

    # Only consider i < j to avoid double counting and self
    iu = np.triu_indices(n, k=1)
    pair_dist = dist[iu]
    pair_min = min_dist[iu]
    # Violation: ri + rj - dist > 0
    pair_viol = np.maximum(0, pair_min - pair_dist)

    total_viol = np.sum(bnd_viol) + np.sum(pair_viol)
    sum_r = np.sum(r)

    return -sum_r, total_viol


def _objective(params):
    val, viol = _compute_objective(params)
    # Penalize violations heavily
    penalty = 1e4 * viol
    return val + penalty


def _gradient(params):
    """
    Numerical gradient via central differences (vectorized where possible).
    For speed, use forward differences with small step.
    """
    n = N
    eps = 1e-7
    grad = np.zeros_like(params)
    base_val, base_viol = _compute_objective(params)
    base_obj = base_val + 1e4 * base_viol

    for i in range(len(params)):
        params[i] += eps
        val, viol = _compute_objective(params)
        obj = val + 1e4 * viol
        params[i] -= eps
        grad[i] = (obj - base_obj) / eps

    return grad


def _smooth_objective(params):
    """
    Smooth surrogate: use soft penalty for violations.
    """
    n = N
    pts = params.reshape(n, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    r = pts[:, 2]

    # Boundary: smooth max
    def soft_max(a, b):
        return np.logaddexp(a, b) - np.log(2)

    # x - r >= 0  =>  -(x - r) <= 0  =>  max(0, r - x)
    # Use softplus for smoothness
    beta = 10.0
    bnd_viol = (1/beta) * np.log1p(np.exp(beta * (r - x))) + \
               (1/beta) * np.log1p(np.exp(beta * (x + r - 1))) + \
               (1/beta) * np.log1p(np.exp(beta * (r - y))) + \
               (1/beta) * np.log1p(np.exp(beta * (x + r - 1) - 0))  # placeholder

    # Actually let's just use the hard penalty with a smooth approximation
    # For now, use the same as _objective but this is the smooth version
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]

    iu = np.triu_indices(n, k=1)
    pair_dist = dist[iu]
    pair_min = min_dist[iu]
    pair_viol = np.maximum(0, pair_min - pair_dist)

    # Boundary violations
    bnd_viol = np.maximum(0, r - x) + np.maximum(0, x + r - 1) + \
               np.maximum(0, r - y) + np.maximum(0, y + r - 1)

    total_viol = np.sum(bnd_viol) + np.sum(pair_viol)
    sum_r = np.sum(r)

    return -sum_r + 1e4 * total_viol


def _hexagonal_init(n):
    """
    Initialize circles in a hexagonal packing pattern.
    """
    # Hexagonal packing: rows offset by half spacing
    # For n circles, arrange in roughly sqrt(n) rows
    rows = int(np.ceil(np.sqrt(n / 1.15)))  # hex is denser
    cols = int(np.ceil(n / rows))

    # Spacing for hexagonal packing
    # If radius is r, horizontal spacing is 2r, vertical is sqrt(3)*r
    # We want to fit in unit square
    # Let's estimate radius from cols and rows
    # Horizontal: cols * 2r <= 1 => r <= 1/(2*cols)
    # Vertical: rows * sqrt(3)*r <= 1 => r <= 1/(sqrt(3)*rows)
    r_est = min(1.0 / (2.0 * cols), 1.0 / (np.sqrt(3) * rows)) * 0.95

    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            x = (j + 0.5) * 2 * r_est
            if i % 2 == 1:
                x += r_est  # offset every other row
            y = i * np.sqrt(3) * r_est + r_est
            circles.append([x, y, r_est])

    # If we don't have enough, add more or adjust
    while len(circles) < n:
        # Add at random positions
        circles.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), r_est * 0.5])

    return np.array(circles[:n]).flatten()


def _grid_init(n):
    """
    Initialize circles in a grid pattern.
    """
    side = int(np.ceil(np.sqrt(n)))
    r = 0.5 / side * 0.95
    circles = []
    for i in range(side):
        for j in range(side):
            if len(circles) >= n:
                break
            circles.append([(i + 0.5) / side, (j + 0.5) / side, r])
    return np.array(circles[:n]).flatten()


def _random_init(n, seed):
    """
    Random initialization.
    """
    rng = np.random.RandomState(seed)
    # Place circles at random positions with small initial radii
    x = rng.uniform(0.1, 0.9, size=n)
    y = rng.uniform(0.1, 0.9, size=n)
    r = rng.uniform(0.02, 0.08, size=n)
    params = np.zeros(3 * n)
    params[0::3] = x
    params[1::3] = y
    params[2::3] = r
    return params


def construct():
    """
    Pack 26 circles in unit square, maximize sum of radii.
    Returns array of shape (26, 3) with (x, y, radius) for each circle.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    best_params = None
    best_score = -np.inf

    # Try multiple initializations
    inits = []

    # Grid initialization
    inits.append(_grid_init(N))

    # Hexagonal initialization
    inits.append(_hexagonal_init(N))

    # Random initializations with different seeds
    for seed in range(9):
        inits.append(_random_init(N, seed=42 + seed))

    # Optimize each initialization
    for init_idx, init_params in enumerate(inits):
        if time.time() > deadline - 5:
            break

        try:
            # Use L-BFGS-B with bounds
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0))  # x
                bounds.append((0.0, 1.0))  # y
                bounds.append((0.0, 0.5))  # r

            # Run optimization
            result = minimize(
                _smooth_objective,
                init_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-8}
            )

            if result.success or result.fun < best_score:
                # Check validity
                _, viol = _compute_objective(result.x)
                if viol < 1e-6:
                    score = -result.fun
                    if score > best_score:
                        best_score = score
                        best_params = result.x.copy()

        except Exception:
            continue

    # If no valid solution found, fall back to grid
    if best_params is None:
        best_params = _grid_init(N)

    # Final polish: try to improve with more iterations
    if time.time() < deadline - 10:
        try:
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0))
                bounds.append((0.0, 1.0))
                bounds.append((0.0, 0.5))

            result = minimize(
                _smooth_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-10}
            )

            _, viol = _compute_objective(result.x)
            if viol < 1e-6:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()

        except Exception:
            pass

    # Convert to (n, 3) array
    result_array = best_params.reshape(N, 3)

    # Ensure circles are inside the unit square (clip if necessary)
    for i in range(N):
        r = result_array[i, 2]
        result_array[i, 0] = np.clip(result_array[i, 0], r, 1 - r)
        result_array[i, 1] = np.clip(result_array[i, 1], r, 1 - r)

    return result_array

# EVOLVE-BLOCK-END
