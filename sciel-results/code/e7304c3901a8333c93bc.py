# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _compute_ratio(points: np.ndarray) -> float:
    """Compute R = (d_min / d_max)^2 for the given points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _penalty_objective(x, lb, mu):
    n = 16
    pts = x.reshape(n, 2)
    dists = pdist(pts)
    d_max = np.max(dists)
    violations = np.maximum(0, lb - dists)
    penalty = np.sum(violations**2)
    return d_max + mu * penalty

def _log_ratio_objective(x):
    n = 16
    pts = x.reshape(n, 2)
    dists = pdist(pts)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_min <= 0 or d_max <= 0:
        return -1e10
    return -(np.log(d_min) - np.log(d_max))

def _solve_with_lower_bound(lb: float) -> tuple:
    """
    Try to find 16 points such that all pairwise distances >= lb.
    Uses penalty method with L-BFGS-B, then polishes with Nelder-Mead on log-ratio.
    Returns (success, points, d_max).
    """
    n = 16
    d = 2
    
    initializations = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Grid
    grid_x = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=float)
    grid_y = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], dtype=float)
    initializations.append(np.column_stack([grid_x, grid_y]))
    
    # 3. Hexagonal lattice subset
    pts_hex = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            pts_hex.append([x, y])
    pts_hex = np.array(pts_hex)
    pts_hex -= pts_hex.mean(axis=0)
    dists_to_center = np.sum(pts_hex**2, axis=1)
    indices = np.argsort(dists_to_center)[:16]
    initializations.append(pts_hex[indices])
    
    # 4. Random
    np.random.seed(123)
    initializations.append(np.random.randn(n, 2) * 2.0)

    best_success = False
    best_points = None
    best_dmax = np.inf
    best_ratio = -1.0

    for x0 in initializations:
        x_flat = x0.flatten()
        
        # Phase 1: Penalty Method
        mu = 1.0
        current_x = x_flat
        
        for _ in range(20):
            try:
                res = minimize(
                    _penalty_objective,
                    current_x,
                    args=(lb, mu),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                current_x = res.x
                pts = current_x.reshape(n, 2)
                dists = pdist(pts)
                d_min = np.min(dists)
                d_max = np.max(dists)
                
                if d_min >= lb - 1e-5:
                    if not best_success or d_max < best_dmax:
                        best_success = True
                        best_points = pts
                        best_dmax = d_max
                    break
                else:
                    mu *= 10
            except Exception:
                break
        
        # Phase 2: Polish with Nelder-Mead on log-ratio
        if current_x is not None:
            try:
                res2 = minimize(
                    _log_ratio_objective,
                    current_x,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                pts_polished = res2.x.reshape(n, 2)
                dists = pdist(pts_polished)
                d_min_p = np.min(dists)
                d_max_p = np.max(dists)
                
                if d_min_p > 1e-6 and d_max_p > 0:
                    ratio_p = (d_min_p / d_max_p) ** 2
                    if ratio_p > best_ratio:
                        best_ratio = ratio_p
                        best_points = pts_polished
                        best_dmax = d_max_p
                        best_success = True
            except Exception:
                pass

    return best_success, best_points, best_dmax

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    d = 2

    best_points = None
    best_ratio = -1.0

    # Sample a range of lb values to guide the penalty method
    lb_values = np.linspace(0.5, 4.0, 15)
    
    for lb in lb_values:
        success, pts, dmax = _solve_with_lower_bound(lb)
        if success and pts is not None:
            ratio = _compute_ratio(pts)
            if ratio > best_ratio:
                best_ratio = ratio
                best_points = pts

    # Fallback
    if best_points is None:
        np.random.seed(42)
        return np.random.randn(n, d)
        
    return best_points
# EVOLVE-BLOCK-END
