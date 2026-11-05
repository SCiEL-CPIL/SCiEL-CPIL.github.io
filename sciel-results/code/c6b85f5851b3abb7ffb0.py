import time
import math
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 145.0

def _log_score(params):
    """Compute log of Delta/n^n. Returns -inf if invalid."""
    pts = params.reshape(8, 2)
    # Compute all pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, 1.0)
    # Check for near-zero distances (distinctness)
    off_diag = dists[~np.eye(8, dtype=bool)]
    if np.any(off_diag < 1e-10):
        return -np.inf
    log_prod = np.sum(np.log(dists))
    max_d = np.max(dists)
    if max_d < 1e-15:
        return -np.inf
    n = 8
    log_prod_scaled = log_prod + n * (n - 1) * math.log(2.0 / max_d)
    log_score = log_prod_scaled - n * math.log(n)
    return log_score

def _neg_log_score(params):
    """Negative log score for minimization."""
    if time.time() > _DEADLINE:
        return 1e10
    ls = _log_score(params)
    if ls < -1e6:
        return 1e10
    return -ls

def _regular_octagon():
    pts = []
    for k in range(8):
        angle = 2 * math.pi * k / 8
        pts.append([math.cos(angle), math.sin(angle)])
    return np.array(pts, dtype=np.float64)

def _two_squares(r1, r2, rot1, rot2):
    """Two concentric squares."""
    pts = []
    for k in range(4):
        angle = rot1 + 2 * math.pi * k / 4
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle)])
    for k in range(4):
        angle = rot2 + 2 * math.pi * k / 4
        pts.append([r2 * math.cos(angle), r2 * math.sin(angle)])
    return np.array(pts, dtype=np.float64)

def _random_initial(scale=1.0, spread=0.5):
    """Generate random points near the unit circle with some spread."""
    angles = 2 * math.pi * np.arange(8) / 8 + np.random.randn(8) * spread
    radii = 1.0 + np.random.randn(8) * 0.3
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _optimize_from(params, method='Nelder-Mead', maxiter=200):
    """Optimize from given parameters, return best result."""
    try:
        if method == 'Nelder-Mead':
            result = minimize(
                _neg_log_score, params,
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
        elif method == 'L-BFGS-B':
            result = minimize(
                _neg_log_score, params,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        else:
            result = minimize(
                _neg_log_score, params,
                method='Powell',
                options={'maxiter': maxiter, 'ftol': 1e-12, 'xtol': 1e-9}
            )
        return result
    except Exception:
        return None

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 145.0
    
    np.random.seed(42)
    
    best_params = _regular_octagon().flatten()
    best_log_score = _log_score(best_params)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular octagon
    initial_configs.append(_regular_octagon())
    
    # Two concentric squares with various parameters
    for r1 in [0.2, 0.4, 0.6, 0.8]:
        for r2 in [1.0, 1.2, 1.4, 1.6]:
            for rot1 in [0.0, math.pi/8, math.pi/4]:
                for rot2 in [0.0, math.pi/8, math.pi/4]:
                    initial_configs.append(_two_squares(r1, r2, rot1, rot2))
    
    # Random perturbations of octagon
    for _ in range(10):
        initial_configs.append(_random_initial(spread=0.3))
    
    # More spread random
    for _ in range(10):
        initial_configs.append(_random_initial(spread=0.7))
    
    # Square + 4 random
    for _ in range(5):
        pts = []
        for k in range(4):
            angle = math.pi/4 + 2 * math.pi * k / 4
            pts.append([math.cos(angle), math.sin(angle)])
        for k in range(4):
            angle = np.random.uniform(0, 2*math.pi)
            r = np.random.uniform(0.3, 1.5)
            pts.append([r * math.cos(angle), r * math.sin(angle)])
        initial_configs.append(np.array(pts, dtype=np.float64))
    
    # Optimize each configuration
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > _DEADLINE:
            break
        
        params = init_pts.flatten()
        
        # Try Nelder-Mead first
        result = _optimize_from(params, method='Nelder-Mead', maxiter=300)
        if result is not None and result.fun < 1e8:
            ls = _log_score(result.x)
            if ls > best_log_score:
                best_log_score = ls
                best_params = result.x.copy()
        
        # Try L-BFGS-B from the best found so far (if time permits)
        if time.time() > _DEADLINE:
            break
        result2 = _optimize_from(best_params, method='L-BFGS-B', maxiter=100)
        if result2 is not None and result2.fun < 1e8:
            ls = _log_score(result2.x)
            if ls > best_log_score:
                best_log_score = ls
                best_params = result2.x.copy()
        
        # Try Powell from best
        if time.time() > _DEADLINE:
            break
        result3 = _optimize_from(best_params, method='Powell', maxiter=50)
        if result3 is not None and result3.fun < 1e8:
            ls = _log_score(result3.x)
            if ls > best_log_score:
                best_log_score = ls
                best_params = result3.x.copy()
    
    # Additional random restarts with Nelder-Mead
    np.random.seed(123)
    for _ in range(15):
        if time.time() > _DEADLINE:
            break
        init_pts = _random_initial(spread=np.random.uniform(0.2, 0.8))
        params = init_pts.flatten()
        result = _optimize_from(params, method='Nelder-Mead', maxiter=200)
        if result is not None and result.fun < 1e8:
            ls = _log_score(result.x)
            if ls > best_log_score:
                best_log_score = ls
                best_params = result.x.copy()
    
    # Final refinement with L-BFGS-B
    if time.time() < _DEADLINE - 5:
        result = _optimize_from(best_params, method='L-BFGS-B', maxiter=200)
        if result is not None and result.fun < 1e8:
            ls = _log_score(result.x)
            if ls > best_log_score:
                best_log_score = ls
                best_params = result.x.copy()
    
    # Convert to list of [x, y]
    final_pts = best_params.reshape(8, 2)
    result_list = []
    for p in final_pts:
        x = float(max(-1e6, min(1e6, p[0])))
        y = float(max(-1e6, min(1e6, p[1])))
        result_list.append([x, y])
    
    return result_list
