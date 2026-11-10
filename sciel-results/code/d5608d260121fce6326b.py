# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

# Precompute all C(13,3) = 286 triangle index combinations
_TRIS = np.array(list(itertools.combinations(range(13), 3)), dtype=np.int32)
_N_TRIS = len(_TRIS)
_I = _TRIS[:, 0]
_J = _TRIS[:, 1]
_K = _TRIS[:, 2]

def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points using precomputed indices."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    # Check for finite points
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 0:
        return 0.0
    
    # Vectorized triangle area calculation using precomputed indices
    p1 = points[_I]
    p2 = points[_J]
    p3 = points[_K]
    
    # Area = 0.5 * |cross product|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull


def _soft_min_score(points: np.ndarray, beta: float = 50.0) -> float:
    """Smooth surrogate: soft-min of triangle areas / hull area.
    Higher is better. Uses log-sum-exp for numerical stability."""
    n = points.shape[0]
    if n < 3:
        return -np.inf
    
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return -np.inf
    
    if A_hull <= 0:
        return -np.inf
    
    p1 = points[_I]
    p2 = points[_J]
    p3 = points[_K]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = np.abs(cross) * 0.5
    
    # Soft-min: -log-sum-exp(-areas/beta) * beta / A_hull
    # This approximates min(areas)/A_hull as beta -> 0
    min_val = np.min(areas)
    if min_val < 1e-15:
        return -np.inf
    
    # For numerical stability, use: soft_min = beta * (min_val - log(sum(exp((min_val - areas)/beta))))
    scaled = (min_val - areas) / beta  # <= 0
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_min = min_val - beta * log_sum
    
    if soft_min <= 0:
        return -np.inf
        
    return soft_min / A_hull


def _negative_soft_min(points_flat: np.ndarray, beta: float = 50.0) -> float:
    """Objective for minimizer: negative soft-min score."""
    points = points_flat.reshape(13, 2)
    s = _soft_min_score(points, beta)
    if np.isneginf(s) or np.isnan(s):
        return 1e6
    return -s


def _local_search(points: np.ndarray, max_iter=20000, initial_step=0.05, time_limit=None) -> np.ndarray:
    """Perform local search to improve the point arrangement."""
    current = points.copy()
    current_score = _score(current)
    n = current.shape[0]
    step_size = initial_step
    start_time = time.time()
    
    for iteration in range(max_iter):
        if time_limit and time.time() > time_limit:
            break
            
        # Pick a random point to move
        idx = np.random.randint(n)
        # Generate a small random perturbation
        delta = np.random.normal(0, step_size, 2)
        new_point = current[idx] + delta
        
        # Keep points within [0, 1]^2
        if new_point[0] < 0 or new_point[0] > 1 or new_point[1] < 0 or new_point[1] > 1:
            continue
        
        candidate = current.copy()
        candidate[idx] = new_point
        candidate_score = _score(candidate)
        
        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score
            step_size = max(step_size * 0.97, 0.001)
        else:
            step_size = min(step_size * 1.03, 0.15)
    
    return current


def _gradient_optimize(points: np.ndarray, time_limit: float, beta_start: float = 200.0, beta_end: float = 10.0) -> np.ndarray:
    """Use L-BFGS-B with smooth soft-min objective for gradient-based optimization."""
    x0 = points.flatten()
    n_params = x0.shape[0]
    
    # Anneal beta from high to low for smooth -> sharp transition
    betas = [beta_start, beta_start * 0.5, beta_start * 0.25, beta_start * 0.1, beta_end]
    
    best_x = x0.copy()
    best_score = _score(points)
    
    for beta in betas:
        if time.time() > time_limit:
            break
        
        res = minimize(
            lambda x, b=beta: _negative_soft_min(x, b),
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1)] * n_params,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if res.success and res.fun < -1e-6:
            candidate = res.x.reshape(13, 2)
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        
        x0 = best_x
    
    return best_x.reshape(13, 2)


def _lattice_search(points: np.ndarray, time_limit: float, grid_res: int = 5) -> np.ndarray:
    """
    Discrete-continuous hybrid search:
    1. Snap points to a grid
    2. Evaluate and keep best
    3. Fine-tune with local search
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Create a grid of candidate positions
    grid = np.linspace(0.05, 0.95, grid_res)
    
    # Try snapping each point to nearby grid positions
    # This is a local refinement around the current best
    for _ in range(500):
        if time.time() > time_limit:
            break
        
        # Pick a random point
        idx = np.random.randint(13)
        
        # Find nearest grid point
        gx = np.clip(np.round((best_points[idx, 0] - 0.05) / (0.9 / (grid_res - 1))), 0, grid_res - 1)
        gy = np.clip(np.round((best_points[idx, 1] - 0.05) / (0.9 / (grid_res - 1))), 0, grid_res - 1)
        
        # Try nearby grid positions (3x3 neighborhood)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                ngx = int(gx) + dx
                ngy = int(gy) + dy
                if 0 <= ngx < grid_res and 0 <= ngy < grid_res:
                    candidate = best_points.copy()
                    candidate[idx, 0] = grid[ngx]
                    candidate[idx, 1] = grid[ngy]
                    s = _score(candidate)
                    if s > best_score:
                        best_score = s
                        best_points = candidate.copy()
        
        # Also try random continuous perturbations
        delta = np.random.normal(0, 0.02, 2)
        candidate = best_points.copy()
        candidate[idx] += delta
        candidate[idx] = np.clip(candidate[idx], 0.01, 0.99)
        s = _score(candidate)
        if s > best_score:
            best_score = s
            best_points = candidate.copy()
    
    return best_points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area.
    
    Uses a multi-phase approach:
    1. Lattice-based discrete search for good initial configurations
    2. Local search refinement
    3. Gradient-based optimization with smooth soft-min objective
    4. Final local refinement
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin before 630s kill
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Generate diverse initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points_reg = np.column_stack((0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)))
    initializations.append(points_reg.copy())
    
    # 2. Grid-like initialization
    grid_pts = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
        [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3],
        [0.2, 0.5], [0.4, 0.5], [0.6, 0.5], [0.8, 0.5]
    ])
    initializations.append(grid_pts)
    
    # 3. Hexagonal pattern
    cx, cy = 0.5, 0.5
    r1 = 0.15
    r2 = 0.35
    hex_center = np.array([[cx, cy]])
    hex_inner = np.array([[cx + r1 * np.cos(2 * np.pi * i / 6), 
                           cy + r1 * np.sin(2 * np.pi * i / 6)] for i in range(6)])
    hex_outer = np.array([[cx + r2 * np.cos(2 * np.pi * (i + 0.5) / 6), 
                           cy + r2 * np.sin(2 * np.pi * (i + 0.5) / 6)] for i in range(6)])
    hex_sym = np.vstack([hex_center, hex_inner, hex_outer])
    hex_sym = np.clip(hex_sym, 0.01, 0.99)
    initializations.append(hex_sym)
    
    # 4. Random points in unit square
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.uniform(0.05, 0.95, (n, 2))
        initializations.append(pts)
    
    # Evaluate all initializations
    for init_pts in initializations:
        s = _score(init_pts)
        if s > best_score:
            best_score = s
            best_points = init_pts.copy()
    
    # Phase 2: Lattice-based discrete search from best start
    if best_points is not None and time.time() < deadline - 120:
        lattice_deadline = min(deadline - 120, start_time + 60)
        lattice_result = _lattice_search(best_points, lattice_deadline, grid_res=6)
        lattice_score = _score(lattice_result)
        if lattice_score > best_score:
            best_score = lattice_score
            best_points = lattice_result.copy()
    
    # Phase 3: Local search refinement
    if best_points is not None and time.time() < deadline - 60:
        ls_deadline = min(deadline - 60, time.time() + 45)
        improved = _local_search(best_points, max_iter=10000, initial_step=0.03, time_limit=ls_deadline)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    # Phase 4: Gradient optimization
    if best_points is not None and time.time() < deadline - 30:
        grad_deadline = min(deadline - 30, time.time() + 50)
        grad_result = _gradient_optimize(best_points, grad_deadline)
        grad_score = _score(grad_result)
        if grad_score > best_score:
            best_score = grad_score
            best_points = grad_result.copy()
    
    # Phase 5: Final local refinement
    if best_points is not None and time.time() < deadline - 5:
        final_deadline = deadline - 5
        final_improved = _local_search(best_points, max_iter=5000, initial_step=0.01, time_limit=final_deadline)
        final_score = _score(final_improved)
        if final_score > best_score:
            best_score = final_score
            best_points = final_improved.copy()
    
    # Final validation
    if best_points is None or best_score <= 0:
        return points_reg
    
    final_score = _score(best_points)
    if final_score <= 0:
        return points_reg
    
    return best_points

# EVOLVE-BLOCK-END
