"""
Packing 50 equal circles in a unit square via differentiable soft-min optimization.
Structural change: use log-sum-exp surrogate for the minimum clearance, optimized
with L-BFGS-B from multiple high-quality initializations, followed by a final
polish phase using incremental distance tracking.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50
HALF = 0.5


def _compute_radius(points):
    """
    Compute the maximum radius such that circles centered at points
    are inside [-0.5, 0.5]^2 and non-overlapping.
    r = min(
        min_{i<j} ||p_i - p_j|| / 2,
        min_i min_k (0.5 - |p_ik|)
    )
    """
    n = len(points)
    if n == 0:
        return 0.0

    # Distance to walls (vectorized)
    wall_dists = np.minimum(HALF - np.abs(points[:, 0]), HALF - np.abs(points[:, 1]))
    min_wall = float(np.min(wall_dists))

    # Minimum pairwise distance via cKDTree
    tree = cKDTree(points)
    # Query nearest neighbors for each point (k=2 to get self + 1 neighbor)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1]))

    r = min(min_pair / 2.0, min_wall)
    return r if r > 0 else 0.0


def _soft_min_clearance(params, beta=50.0):
    """
    Differentiable soft-minimum of all clearances (pairwise distances/2 and wall distances).
    Uses log-sum-exp: soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    This is a smooth lower bound on the true minimum that becomes tight as beta -> inf.
    """
    points = params.reshape(-1, 2)
    n = len(points)
    
    # Wall clearances: for each point, min of (HALF - |x|, HALF - |y|)
    wall_x = HALF - np.abs(points[:, 0])
    wall_y = HALF - np.abs(points[:, 1])
    wall_clears = np.minimum(wall_x, wall_y)
    
    # Pairwise distances / 2
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_clears = dists / 2.0
    
    # Combine all clearances into one vector
    all_clears = np.concatenate([wall_clears, pair_clears.ravel()])
    
    # Soft min via log-sum-exp (numerically stable)
    min_c = np.min(all_clears)
    exp_terms = np.exp(-beta * (all_clears - min_c))
    soft_min = min_c - (1.0 / beta) * np.log(np.sum(exp_terms))
    
    return soft_min


def _soft_min_grad(params, beta=50.0):
    """
    Gradient of the soft-minimum clearance w.r.t. all 2N coordinates.
    d(soft_min)/dp = -sum_j w_j * d(clearance_j)/dp where w_j = exp(-beta*(c_j - c_min)) / sum(exp(...))
    """
    points = params.reshape(-1, 2)
    n = len(points)
    
    # Wall clearances
    wall_x = HALF - np.abs(points[:, 0])
    wall_y = HALF - np.abs(points[:, 1])
    wall_clears = np.minimum(wall_x, wall_y)
    
    # Pairwise distances / 2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_clears = dists / 2.0
    
    all_clears = np.concatenate([wall_clears, pair_clears.ravel()])
    
    # Compute weights
    min_c = np.min(all_clears)
    exp_terms = np.exp(-beta * (all_clears - min_c))
    weights = exp_terms / np.sum(exp_terms)
    
    # Gradient of wall clearances w.r.t. points
    # d(wall_x_i)/dx_i = -sign(x_i), d(wall_y_i)/dy_i = -sign(y_i)
    grad = np.zeros_like(points)
    
    # Wall contributions
    for i in range(n):
        w = weights[i]
        if wall_x[i] <= wall_y[i]:
            grad[i, 0] += -w * np.sign(points[i, 0])
        else:
            grad[i, 1] += -w * np.sign(points[i, 1])
    
    # Pairwise contributions
    # d(pair_clear_{ij})/dp_i = (p_i - p_j) / dist_{ij}  (since pair_clear = dist/2, derivative is (p_i-p_j)/dist)
    # Actually: d(dist_{ij}/2)/dp_i = (p_i - p_j) / dist_{ij}
    # The flat index for pair (i,j) where i < j is: i*n - i*(i+1)/2 + j - i - 1
    # Let's use a simpler approach: accumulate contributions
    for i in range(n):
        for j in range(i + 1, n):
            w = weights[n + i * n + j]  # flat index in pair_clears.ravel()
            if dists[i, j] > 1e-12:
                direction = (points[i] - points[j]) / dists[i, j]
                grad[i] += w * direction
                grad[j] += -w * direction
    
    return grad.ravel()


def _neg_soft_min(params, beta=50.0):
    """Objective: minimize negative soft-min clearance (i.e., maximize it)."""
    points = params.reshape(-1, 2)
    if np.any(np.abs(points) > HALF + 1e-10):
        return 1e10
    if not np.all(np.isfinite(points)):
        return 1e10
    return -_soft_min_clearance(params, beta)


def _neg_soft_min_grad(params, beta=50.0):
    """Gradient of negative soft-min clearance."""
    points = params.reshape(-1, 2)
    if np.any(np.abs(points) > HALF + 1e-10):
        return np.zeros_like(params)
    if not np.all(np.isfinite(points)):
        return np.zeros_like(params)
    return -_soft_min_grad(params, beta)


def _hex_initial(seed=42):
    """
    Hexagonal lattice initial placement with spacing tuned for 50 circles.
    The optimal radius is ~0.0714, so spacing should be ~2*r = 0.1428.
    We use a hexagonal grid that fits 50 points in the unit square.
    """
    rng = np.random.RandomState(seed)
    
    # Target spacing based on known optimal radius
    target_r = 0.0714
    spacing = 2.0 * target_r  # ~0.1428
    
    pts = []
    row = 0
    max_rows = 20
    while len(pts) < N and row < max_rows:
        y = -HALF + spacing * (row + 0.5)
        if y > HALF - 0.01:
            break
        offset = 0 if row % 2 == 0 else spacing * 0.5
        x = -HALF + spacing * 0.5 + offset
        col = 0
        while x < HALF - 0.01 and len(pts) < N:
            pts.append([x, y])
            x += spacing
            col += 1
        row += 1
    
    if len(pts) < N:
        # If we didn't get enough points, add more rows or adjust
        # Fall back to a denser grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = grid[:N].tolist()
    
    pts = np.array(pts[:N])
    # Small jitter to break symmetry
    pts += rng.uniform(-0.003, 0.003, size=pts.shape)
    pts = np.clip(pts, -HALF + 0.01, HALF - 0.01)
    return pts


def _grid_init():
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_init(seed=999):
    """Random initialization inside the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _lbfgs_optimize(points, deadline, beta_start=20.0, beta_end=200.0):
    """
    Optimize circle centers using L-BFGS-B on the soft-min clearance objective.
    Uses annealing of beta (soft-min sharpness) to first find a good region,
    then refine with a sharper approximation of the true minimum.
    """
    current = points.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Define bounds
    bounds = [(-HALF + 1e-12, HALF - 1e-12)] * (2 * N)
    
    # Anneal beta from low to high
    num_phases = 5
    betas = np.geomspace(beta_start, beta_end, num_phases)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        # Run L-BFGS-B with current beta
        try:
            result = minimize(
                _neg_soft_min,
                current.ravel(),
                jac=_neg_soft_min_grad,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
            )
            if result.success or result.fun < 1e8:
                new_points = result.x.reshape(-1, 2)
                new_points = np.clip(new_points, -HALF + 1e-12, HALF - 1e-12)
                new_r = _compute_radius(new_points)
                if new_r > best_r:
                    best_r = new_r
                    best = new_points.copy()
                current = new_points
        except Exception:
            pass
    
    return best, best_r


def _local_polish(points, deadline=None):
    """
    Fast local refinement: iteratively push the most constrained points apart.
    Uses incremental updates for speed.
    """
    rng = np.random.RandomState(777)
    current = points.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r

    n = len(current)
    step_size = 0.001
    max_iter = 30000

    # Precompute distance matrix for incremental updates
    def _update_dist_matrix(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        return dists

    dist_mat = _update_dist_matrix(current)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Find the point with the smallest clearance
        wall_dists = np.minimum(HALF - np.abs(current[:, 0]), HALF - np.abs(current[:, 1]))
        np.fill_diagonal(dist_mat, np.inf)
        min_pair_dists = np.min(dist_mat, axis=1)
        clearances = np.minimum(wall_dists, min_pair_dists / 2.0)
        worst_idx = int(np.argmin(clearances))

        # Try to move the worst point in a random direction
        direction = rng.normal(0, 1, size=2)
        direction /= (np.linalg.norm(direction) + 1e-15)

        # Try several step sizes
        improved = False
        for ss in [step_size, step_size * 0.5, step_size * 2.0, step_size * 0.25]:
            trial = current.copy()
            trial[worst_idx] = current[worst_idx] + direction * ss
            trial[worst_idx] = np.clip(trial[worst_idx], -HALF + 1e-12, HALF - 1e-12)
            trial_r = _compute_radius(trial)
            if trial_r > current_r + 1e-15:
                current = trial
                current_r = trial_r
                # Update distance matrix
                dist_mat = _update_dist_matrix(current)
                if current_r > best_r:
                    best_r = current_r
                    best = current.copy()
                improved = True
                break

        if not improved:
            # Try moving in opposite direction
            for ss in [step_size, step_size * 0.5]:
                trial = current.copy()
                trial[worst_idx] = current[worst_idx] - direction * ss
                trial[worst_idx] = np.clip(trial[worst_idx], -HALF + 1e-12, HALF - 1e-12)
                trial_r = _compute_radius(trial)
                if trial_r > current_r + 1e-15:
                    current = trial
                    current_r = trial_r
                    dist_mat = _update_dist_matrix(current)
                    if current_r > best_r:
                        best_r = current_r
                        best = current.copy()
                    break

    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Multiple initializations: hex lattice (best), grid, random
    inits = [
        _hex_initial(seed=42),
        _hex_initial(seed=123),
        _grid_init(),
        _random_init(seed=999),
        _random_init(seed=555),
    ]

    best_points = None
    best_r = 0.0

    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break

        # Phase 1: L-BFGS-B optimization with soft-min
        lbfgs_points, lbfgs_r = _lbfgs_optimize(
            init,
            deadline=deadline
        )
        if lbfgs_r > best_r:
            best_r = lbfgs_r
            best_points = lbfgs_points

        # Phase 2: Local polish
        if time.time() > deadline:
            break
        polish_points, polish_r = _local_polish(
            lbfgs_points,
            deadline=deadline
        )
        if polish_r > best_r:
            best_r = polish_r
            best_points = polish_points

    # Fallback
    if best_points is None:
        best_points = _grid_init()
        best_r = _compute_radius(best_points)

    # Final validation
    best_points = np.clip(best_points, -HALF + 1e-12, HALF - 1e-12)
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        best_points = _grid_init()

    return best_points


# EVOLVE-BLOCK-START
# (The above construct function is the entry point)
# EVOLVE-BLOCK-END
