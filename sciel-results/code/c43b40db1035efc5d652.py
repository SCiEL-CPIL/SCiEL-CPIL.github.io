"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

START_TIME = time.time()
TIME_LIMIT = 115.0  # Leave 5s margin for the 120s limit


def _compute_radius(c):
    """Compute the maximum radius for given centers."""
    # Distance to boundaries
    boundary_dist = 0.5 - np.abs(c).max(axis=1)
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = d.min()
    return min(boundary_dist.min(), min_pair_dist / 2.0)


def _softmin_neg(c, beta=1000.0):
    """Negative soft-min of constraints (pairwise and boundary)."""
    c = c.reshape(-1, 2)
    # Boundary constraints
    boundary = 0.5 - np.abs(c).max(axis=1)
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Use soft-min approximation: -log-sum-exp
    all_vals = np.concatenate([boundary, min_pair / 2.0 * np.ones(1)])
    # Soft min: approximate min by -1/beta * log(sum(exp(-beta * x)))
    max_val = all_vals.max()
    soft_min = max_val - np.log(np.sum(np.exp(-beta * (all_vals - max_val)))) / beta
    return -soft_min


def _grad_softmin_neg(c, beta=1000.0):
    """Gradient of negative soft-min."""
    c = c.reshape(-1, 2)
    # For simplicity, use finite differences in the optimizer
    return None


def _initial_hex_grid(n):
    """Generate a hexagonal grid of n points in [-0.5, 0.5]^2."""
    # Hexagonal packing: rows offset
    points = []
    # Estimate how many rows/columns we need
    # For hex packing, density is higher than square
    # Try to fit about sqrt(n) rows
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    
    # Spacing in hex grid
    # For n points in unit square, spacing ~ 1/sqrt(n)
    # Hex: horizontal spacing s, vertical spacing s*sqrt(3)/2
    # Let's estimate s
    s = 1.0 / np.sqrt(n) * 1.1  # slightly larger to fit in square
    
    for i in range(rows + 10):
        row_y = -0.5 + (i + 0.5) * s * np.sqrt(3) / 2
        if row_y > 0.5:
            break
        offset = s / 2 if i % 2 == 0 else 0
        cols = int(np.floor(1.0 / s)) + 1
        for j in range(cols + 10):
            row_x = -0.5 + (j + 0.5) * s + offset
            if -0.5 <= row_x <= 0.5 and -0.5 <= row_y <= 0.5:
                points.append([row_x, row_y])
    
    points = np.array(points)
    
    # If we have too many, randomly subsample
    if len(points) > n:
        indices = np.random.choice(len(points), n, replace=False)
        points = points[indices]
    
    # If we have too few, pad with random points
    while len(points) < n:
        points = np.vstack([points, np.random.uniform(-0.45, 0.45, size=(1, 2))])
    
    return points[:n]


def _initial_square_grid(n):
    """Generate a square grid of n points in [-0.5, 0.5]^2."""
    m = int(np.ceil(np.sqrt(n)))
    # Adjust to get closer to n
    while m * m < n:
        m += 1
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _refine(c, time_budget=30.0):
    """Refine positions using L-BFGS-B with smooth objective."""
    best_c = c.copy()
    best_r = _compute_radius(c)
    start = time.time()
    
    beta = 500.0
    for _ in range(5):  # Try a few different beta values
        if time.time() - start > time_budget:
            break
        
        def obj(x):
            return _softmin_neg(x, beta)
        
        try:
            res = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            new_c = res.x.reshape(-1, 2)
            new_r = _compute_radius(new_c)
            if new_r > best_r:
                best_r = new_r
                best_c = new_c
        except Exception:
            pass
        
        beta *= 2  # Increase sharpness
    
    return best_c, best_r


def _refine_local(c, time_budget=20.0):
    """Local refinement using coordinate descent / steepest ascent."""
    best_c = c.copy()
    best_r = _compute_radius(c)
    start = time.time()
    
    step_size = 0.001
    for iteration in range(5000):
        if time.time() - start > time_budget:
            break
        
        # Try small random perturbations
        c_trial = best_c.copy()
        # Move a few random points
        n_move = min(5, N)
        idx = np.random.choice(N, n_move, replace=False)
        c_trial[idx] += np.random.normal(0, step_size, size=(n_move, 2))
        # Clamp to square
        c_trial = np.clip(c_trial, -0.5 + 1e-10, 0.5 - 1e-10)
        
        r_trial = _compute_radius(c_trial)
        if r_trial > best_r:
            best_r = r_trial
            best_c = c_trial
            step_size *= 1.1
        else:
            step_size *= 0.9
            step_size = max(step_size, 1e-6)
    
    return best_c, best_r


def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing minimum radius."""
    global START_TIME, TIME_LIMIT
    
    # Generate multiple initial candidates
    candidates = []
    
    # Square grid
    c_square = _initial_square_grid(N)
    candidates.append(c_square)
    
    # Hex grid
    np.random.seed(42)
    c_hex = _initial_hex_grid(N)
    candidates.append(c_hex)
    
    # Slightly perturbed square grid
    np.random.seed(123)
    c_perturbed = c_square + np.random.normal(0, 0.01, size=(N, 2))
    c_perturbed = np.clip(c_perturbed, -0.49, 0.49)
    candidates.append(c_perturbed)
    
    # Evaluate initial candidates
    best_c = None
    best_r = 0
    for c in candidates:
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c
    
    # Refine each candidate with time budget
    elapsed = time.time() - START_TIME
    time_remaining = TIME_LIMIT - elapsed - 10  # 10s safety margin
    
    if time_remaining > 10:
        time_per_candidate = time_remaining / len(candidates)
        for i, c in enumerate(candidates):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            c_refined, r_refined = _refine(c, time_budget=time_per_candidate)
            if r_refined > best_r:
                best_r = r_refined
                best_c = c_refined
    
    # Final local refinement
    elapsed = time.time() - START_TIME
    time_remaining = TIME_LIMIT - elapsed - 5
    if time_remaining > 5:
        c_final, r_final = _refine_local(best_c, time_budget=time_remaining)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final
    
    # Ensure valid output
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Final check
    if not np.all(np.isfinite(best_c)):
        return _initial_square_grid(N)
    
    return best_c
# EVOLVE-BLOCK-END
