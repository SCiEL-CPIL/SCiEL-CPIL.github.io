# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _objective_full(points_flat):
    """Objective for full 32-dim optimization."""
    n = 16
    points = points_flat.reshape(n, 2)
    score = _compute_score(points)
    if score <= 0:
        return 1e10
    return -np.log(score)


def _objective_reduced(x):
    """
    Objective for reduced 28-dim optimization.
    Points 0 and 1 are fixed at (0,0) and (1,0) to break symmetries.
    x is a flat array of length 28 representing points 2..15.
    """
    n = 16
    free_points = x.reshape(14, 2)
    points = np.zeros((n, 2))
    points[0] = [0.0, 0.0]
    points[1] = [1.0, 0.0]
    points[2:] = free_points
    
    score = _compute_score(points)
    if score <= 0:
        return 1e10
    return -np.log(score)


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit

    best_points = None
    best_score = 0.0

    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # --- Phase 1: Local Optimization from Structured Initializations ---
    # We use the full 32-dim space here with L-BFGS-B for fast convergence from good starts.
    
    initial_configs = []

    # 1. 4x4 grid
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    initial_configs.append(grid)

    # 2. Centered 4x4 grid
    grid2 = np.array([[i-1.5, j-1.5] for i in range(4) for j in range(4)], dtype=float)
    initial_configs.append(grid2 * 1.5)

    # 3. Hexagonal-ish packing
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts, dtype=float)
    initial_configs.append(hex_pts)

    # 4. Two concentric squares
    inner = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    outer = []
    # 12 points on a larger square perimeter? 
    # Let's just use a random set for now, or a specific structured one.
    # A 4x4 grid is already covered. Let's try a circle-based one.
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(circle_pts)

    # 5. Random starts
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.standard_normal((n, d))
        initial_configs.append(pts)

    for config in initial_configs:
        if time.time() > deadline - 120:  # Leave time for Phase 2
            break

        flat = config.flatten()
        try:
            result = minimize(
                _objective_full,
                flat,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < 1e9:  # Valid result
                opt_points = result.x.reshape(n, 2)
                opt_score = _compute_score(opt_points)
                update_best(opt_points, opt_score)
        except Exception:
            continue

    # --- Phase 2: Global Search with Reduced Dimensionality ---
    # Use differential evolution on the 28-dim reduced space.
    # This is slower per evaluation but better for escaping local optima.
    
    n_free = 14
    bounds = [(-5.0, 5.0)] * (n_free * 2)
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s buffer
            break
            
        try:
            res = differential_evolution(
                _objective_reduced, 
                bounds, 
                seed=seed, 
                maxiter=200, 
                tol=1e-8,
                polish=True,
                init='latinhypercube',
                popsize=15
            )
            
            # Reconstruct full points from reduced result
            free_points = res.x.reshape((n_free, 2))
            points = np.zeros((n, 2))
            points[0] = [0.0, 0.0]
            points[1] = [1.0, 0.0]
            points[2:] = free_points
            
            score = _compute_score(points)
            update_best(points, score)
            
        except Exception:
            continue

    # --- Phase 3: Final Polish ---
    # If we have time, run a few more local optimizations from the best found so far
    # or random perturbations of the best found.
    
    if best_points is not None and time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb best points
            noise = np.random.default_rng().normal(0, 0.1, best_points.shape)
            perturbed = best_points + noise
            flat = perturbed.flatten()
            try:
                result = minimize(
                    _objective_full,
                    flat,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < 1e9:
                    opt_points = result.x.reshape(n, 2)
                    opt_score = _compute_score(opt_points)
                    update_best(opt_points, opt_score)
            except Exception:
                continue

    # Fallback if no valid points were found
    if best_points is None:
        best_points = initial_configs[0]

    return best_points


# EVOLVE-BLOCK-END
