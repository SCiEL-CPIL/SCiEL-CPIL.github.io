# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

def _compute_areas(points: np.ndarray, i_sel: np.ndarray, j_sel: np.ndarray, k_sel: np.ndarray) -> np.ndarray:
    """Compute areas for all triangles defined by indices."""
    p1 = points[i_sel]
    p2 = points[j_sel]
    p3 = points[k_sel]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return np.abs(cross) * 0.5

def _score(points: np.ndarray) -> float:
    """Compute A_min / A_hull for a set of points."""
    n = points.shape[0]
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
    
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 0.0
    
    if A_hull <= 1e-12:
        return 0.0
    
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    areas = _compute_areas(points, i_sel, j_sel, k_sel)
    min_area = np.min(areas)
    
    if min_area < 1e-12:
        return 0.0
        
    return min_area / A_hull

def _soft_min_objective(x: np.ndarray, i_sel: np.ndarray, j_sel: np.ndarray, k_sel: np.ndarray, tau: float = 50.0) -> float:
    """
    Objective function for minimization.
    Minimizes - (A_min / A_hull).
    Uses a soft-min approximation for A_min to ensure differentiability.
    A_min_soft = (1/tau) * ln(sum(exp(-tau * A_i)))  is actually a soft-max for -A_i?
    Let's check:
    We want to approximate min(A_i).
    log-sum-exp trick: min(A_i) approx - (1/tau) * ln(sum(exp(-tau * A_i))).
    So A_min_soft = - (1/tau) * ln(sum(exp(-tau * A_i))).
    We want to maximize A_min / A_hull.
    So we minimize - (A_min / A_hull).
    Obj = - A_min_soft / A_hull.
    """
    points = x.reshape(-1, 2)
    
    # Check validity
    if not np.all(np.isfinite(points)):
        return 1e10
        
    try:
        hull = ConvexHull(points)
        A_hull = hull.volume
    except Exception:
        return 1e10
        
    if A_hull <= 1e-12:
        return 1e10
        
    areas = _compute_areas(points, i_sel, j_sel, k_sel)
    
    # Handle zero areas in exp to avoid inf
    areas_clipped = np.clip(areas, 1e-12, None)
    
    # Soft min: min(A) approx -1/tau * log(sum(exp(-tau*A)))
    # To avoid overflow in exp(-tau * A), we can shift by max, but here A is small.
    # Actually, exp(-tau * A) is small if A is large.
    # If A is very small (near 0), exp(-tau*A) is near 1.
    # sum(exp(-tau*A)) is roughly N.
    # -1/tau * log(N) is negative constant? No.
    # Let's use the standard stable soft-min.
    # min(x) = -1/tau * log(sum(exp(-tau * x)))
    # This is correct.
    
    # However, if A_i are very small, -tau*A_i is near 0, exp is near 1.
    # If A_i are large, -tau*A_i is large negative, exp is 0.
    # So the sum is dominated by the smallest A_i.
    
    exponents = -tau * areas_clipped
    # Shift for stability if exponents are large positive (which they aren't, they are <= 0)
    # So exp(exponents) is in (0, 1].
    sum_exp = np.sum(np.exp(exponents))
    
    if sum_exp == 0:
        return 1e10
        
    A_min_soft = - (1.0 / tau) * np.log(sum_exp)
    
    # We want to maximize A_min_soft / A_hull
    # So minimize - (A_min_soft / A_hull)
    return - (A_min_soft / A_hull)

def _gradient_optimize(points: np.ndarray, i_sel: np.ndarray, j_sel: np.ndarray, k_sel: np.ndarray, time_limit: float) -> np.ndarray:
    """
    Perform gradient-based optimization using L-BFGS-B.
    """
    n = points.shape[0]
    x0 = points.flatten()
    
    # Bounds: keep points in [0.01, 0.99] to avoid boundary issues with ConvexHull
    bounds = [(0.01, 0.99)] * (2 * n)
    
    start_time = time.time()
    
    def objective(x):
        if time.time() > time_limit:
            return 1e10
        return _soft_min_objective(x, i_sel, j_sel, k_sel, tau=50.0)
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if result.success or result.fun < objective(x0):
            best_x = result.x
        else:
            best_x = x0
            
    except Exception:
        best_x = x0
        
    return best_x.reshape(n, 2)

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the smallest triangle area ratio.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Precompute triangle indices
    idx = np.arange(n)
    i, j, k = np.meshgrid(idx, idx, idx, indexing='ij')
    i_flat = i.flatten()
    j_flat = j.flatten()
    k_flat = k.flatten()
    mask = (i_flat < j_flat) & (j_flat < k_flat)
    i_sel = i_flat[mask]
    j_sel = j_flat[mask]
    k_sel = k_flat[mask]
    
    best_points = None
    best_score = -1.0
    
    # Initial configurations
    # 1. Grid-like
    grid = np.array([[i/3, j/3] for i in range(4) for j in range(4) if not (i==3 and j==3)])
    grid = np.clip(grid, 0.05, 0.95)
    
    # 2. Hexagonal packing (approx)
    hex_pts = np.array([
        [0.2, 0.2], [0.5, 0.2], [0.8, 0.2],
        [0.35, 0.45], [0.65, 0.45],
        [0.2, 0.7], [0.5, 0.7], [0.8, 0.7],
        [0.35, 0.9], [0.65, 0.9],
        [0.5, 0.35], [0.35, 0.6], [0.65, 0.6]
    ])
    
    # 3. Regular polygon + center
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    radius = 0.4
    poly_pts = np.column_stack([0.5 + radius*np.cos(angles), 0.5 + radius*np.sin(angles)])
    poly_pts = np.vstack([poly_pts, [[0.5, 0.5]]])
    
    # 4. Random seeds
    rng = np.random.default_rng(42)
    random_starts = [rng.random((13, 2)) * 0.9 + 0.05 for _ in range(10)]
    
    starts = [grid, hex_pts, poly_pts] + random_starts
    
    # Evaluate starts
    for pts in starts:
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Optimize from best start
    if best_points is not None:
        # Allocate 50% of time to the best initial guess
        alloc_time = time.time() + (deadline - time.time()) * 0.5
        improved = _gradient_optimize(best_points, i_sel, j_sel, k_sel, time_limit=alloc_time)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
    
    # Random restarts with optimization
    while time.time() < deadline - 20:
        pts = rng.random((13, 2)) * 0.9 + 0.05
        
        # Quick check initial score
        score = _score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
        # Allocate remaining time per restart
        remaining = deadline - 10 - time.time()
        if remaining < 10:
            break
            
        # Limit time per restart to ensure we try multiple
        restart_deadline = time.time() + min(remaining, 20)
        
        improved = _gradient_optimize(pts, i_sel, j_sel, k_sel, time_limit=restart_deadline)
        improved_score = _score(improved)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved.copy()
            
        # If we found a very good score, we might want to polish it with more time?
        # L-BFGS-B is fast, so we can just keep restarting.
        
    if best_points is None:
        best_points = rng.random((13, 2)) * 0.9 + 0.05
        
    return best_points

# EVOLVE-BLOCK-END
