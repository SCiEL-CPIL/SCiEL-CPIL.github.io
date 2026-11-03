"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for efficiency
_idx_i, _idx_j, _idx_k = np.indices((N, N, N)).reshape(3, -1)
_mask = (_idx_i < _idx_j) & (_idx_j < _idx_k)
_TRI_I = _idx_i[_mask]
_TRI_J = _idx_j[_mask]
_TRI_K = _idx_k[_mask]

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _logsumexp_min_obj(params, tau):
    """
    Smooth surrogate for the negative minimum triangle area ratio.
    params: (2N,) array of [x_1, y_1, ..., x_N, y_N]
    tau: sharpness parameter for logsumexp approximation of min
    """
    pts = params.reshape(N, 2)
    
    # Compute Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 1e9
    except Exception:
        return 1e9

    # Compute all triangle areas using pre-computed indices
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    ratios = areas / hull_area
    
    # We want to maximize min(ratios).
    # Minimize -min(ratios).
    # Approx -min(y) with (1/tau) * log(sum(exp(-tau * y)))
    # Note: for large tau, this converges to -min(y).
    
    # Numerical stability: subtract max of exponents
    exponents = -tau * ratios
    max_exp = np.max(exponents)
    if max_exp > 100:
        # Overflow protection, though ratios are small
        max_exp = 100
    
    sum_exp = np.sum(np.exp(exponents - max_exp))
    log_sum = max_exp + np.log(sum_exp)
    
    return log_sum / tau

def _smooth_optimize(p0, time_limit):
    """
    Optimize using BFGS on the smooth logsumexp objective with annealing.
    """
    start_time = time.time()
    
    # Annealing schedule for tau - more granular for better convergence
    taus = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
    
    current_pts = p0.copy()
    best_score = _compute_score(current_pts)
    
    for tau in taus:
        if (time.time() - start_time) > time_limit:
            break
            
        def objective(params):
            return _logsumexp_min_obj(params, tau)
        
        try:
            res = minimize(
                objective, 
                current_pts.flatten(), 
                method='BFGS', 
                options={'maxiter': 200, 'gtol': 1e-8, 'factr': 1e10}
            )
            
            new_pts = res.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                current_pts = new_pts.copy()
                
        except Exception:
            continue
            
    # Final high-precision polish with very high tau
    if time_limit - (time.time() - start_time) > 1.0:
        try:
            def objective_polish(params):
                return _logsumexp_min_obj(params, 1000.0)
            
            res = minimize(
                objective_polish, 
                current_pts.flatten(), 
                method='BFGS', 
                options={'maxiter': 300, 'gtol': 1e-10, 'factr': 1e12}
            )
            
            new_pts = res.x.reshape(N, 2)
            new_score = _compute_score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                current_pts = new_pts.copy()
                
        except Exception:
            pass
            
    return current_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    # Generate diverse starts
    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    
    starts = [p_reg]
    
    # 2. Jittered regular polygons with varying noise levels
    for noise_scale in [0.05, 0.1, 0.15, 0.2, 0.3]:
        noise = rng.normal(0, noise_scale, size=(N, 2))
        p_jit = p_reg + noise
        starts.append(p_jit)
        
    # 3. Random convex-like starts
    for _ in range(8):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.7, 1.3, N)
        p_rand = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        starts.append(p_rand)
        
    # 4. Halton sequence for better space-filling
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = np.array([[halton(i+1, 2), halton(i+1, 3)] for i in range(N)])
    # Normalize to unit circle
    halton_pts = (halton_pts - 0.5) * 2.0
    starts.append(halton_pts)
    
    # Optimize each start with adaptive time allocation
    for idx, p0 in enumerate(starts):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3.0:
            break
            
        # Allocate more time to early promising starts
        # First 3 starts get 60% of remaining, rest get 40%
        if idx < 3:
            alloc_time = remaining * 0.6
        else:
            alloc_time = remaining * 0.4 / max(1, len(starts) - idx)
            
        try:
            pts_opt, score_opt = _smooth_optimize(p0, alloc_time)
            
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt.copy()
                
        except Exception:
            continue
            
    if best_pts is None:
        return p_reg
        
    return best_pts

# EVOLVE-BLOCK-END
