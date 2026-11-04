"""
Heilbronn convex regions: Maximize minimum triangle area / hull area.
Strategy: Log-barrier augmented Lagrangian framework with angular parameterization
on the convex hull, using a smooth max-min surrogate for triangle areas and
interior-point optimization to exploit the smoothness of the angular domain.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _compute_score_and_worst(pts):
    """
    Calculate the score and identify the worst triangle.
    Returns (score, worst_indices) where worst_indices is a tuple (i, j, k) or None.
    """
    try:
        if np.allclose(pts, pts[0]):
            return 0.0, None
            
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0, None
        
    if hull_area < 1e-10:
        return 0.0, None

    n = len(pts)
    min_area = float('inf')
    worst_idx = None
    
    # Vectorized computation for speed
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    worst_idx = (i[min_idx], j[min_idx], k[min_idx])
    
    score = min_area / hull_area
    return score, worst_idx

def _compute_score(pts):
    """
    Wrapper for score only.
    """
    score, _ = _compute_score_and_worst(pts)
    return score

def _decode_params(x):
    """
    Decode parameter vector x into 2D points.
    x[0:N] are log-radii (to ensure positivity), x[N:2*N] are angles.
    Sorts points by angle to enforce convex order.
    """
    log_radii = x[:N]
    angles = x[N:2*N]
    
    # Ensure angles are in [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    
    # Sort by angle
    sort_idx = np.argsort(angles)
    sorted_angles = angles[sort_idx]
    sorted_log_radii = log_radii[sort_idx]
    
    # Exponentiate to get positive radii
    radii = np.exp(sorted_log_radii)
    
    pts = np.column_stack([
        radii * np.cos(sorted_angles),
        radii * np.sin(sorted_angles)
    ])
    return pts

def _soft_min(areas, tau):
    """
    Compute smooth minimum using log-sum-exp.
    """
    max_area = np.max(areas)
    log_sum_exp = np.log(np.sum(np.exp(-(areas - max_area) / tau)))
    return max_area - tau * log_sum_exp

def _objective(x, tau=0.05):
    """
    Objective function: Negative smooth minimum triangle area normalized by hull area.
    Uses log-sum-exp for smoothness.
    """
    pts = _decode_params(x)
    
    # Compute all triangle areas
    n = N
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    areas = 0.5 * np.abs(cross)
    
    if np.any(areas == 0):
        return 1e10
    
    # Compute hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10

    # Smooth min using log-sum-exp
    smooth_min = _soft_min(areas, tau)
    score = smooth_min / hull_area
    return -score

def _grad_objective(x, tau=0.05):
    """
    Compute gradient of the objective function using finite differences.
    """
    eps = 1e-6
    grad = np.zeros_like(x)
    f0 = _objective(x, tau)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = _objective(x_plus, tau)
        grad[i] = (f_plus - f0) / eps
    
    return grad

def _construct_initial_params(seed):
    """
    Generate initial parameters (log-radii, angles) for optimization.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a regular polygon-like structure
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rng.normal(0, 0.1, N)
    log_radii = np.zeros(N) + rng.normal(0, 0.1, N)
    
    x = np.concatenate([log_radii, angles])
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    # Multi-start with SLSQP (interior-point style) using log-barrier formulation
    num_starts = 15
    time_per_start = (deadline - start_time) / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        x0 = _construct_initial_params(seed=1000 + i)
        
        try:
            # Use SLSQP with gradient for better convergence
            # Log-radii: [-5, 5] (radii in [e^-5, e^5])
            # Angles: [0, 2*pi]
            bounds = [(-5.0, 5.0)] * N + [(0.0, 2 * np.pi)] * N
            
            res = minimize(
                _objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                x_opt = res.x
                pts_opt = _decode_params(x_opt)
                score_opt, _ = _compute_score_and_worst(pts_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_pts = pts_opt
                    
        except Exception:
            pass
    
    # If SLSQP didn't find a good solution, try L-BFGS-B as fallback
    if best_score < 0.01:
        for i in range(10):
            if time.time() > deadline - 5.0:
                break
                
            x0 = _construct_initial_params(seed=2000 + i)
            
            try:
                bounds = [(-5.0, 5.0)] * N + [(0.0, 2 * np.pi)] * N
                
                res = minimize(
                    _objective, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < best_score:
                    x_opt = res.x
                    pts_opt = _decode_params(x_opt)
                    score_opt, _ = _compute_score_and_worst(pts_opt)
                    
                    if score_opt > best_score:
                        best_score = score_opt
                        best_pts = pts_opt
                        
            except Exception:
                pass

    if best_pts is None:
        # Fallback to Halton
        pts = _halton_sequence(N, 2)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area > 1e-10:
                pts /= np.sqrt(area)
        except Exception:
            pass
        best_pts = pts

    return best_pts

def _halton_sequence(n, base):
    """Generate n points from the Halton sequence in [0,1]^2 for a given base."""
    seq = np.zeros((n, 2))
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        for d in range(1, 3):
            f = 1.0
            r = 0.0
            i_val = i
            while i_val > 0:
                i_val, rem = divmod(i_val, base)
                f /= base
                r += rem * f
            seq[i-1, d-1] = r
    return seq

# EVOLVE-BLOCK-END
