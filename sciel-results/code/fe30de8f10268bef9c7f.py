import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx, j_idx, k_idx = i_idx[mask], j_idx[mask], k_idx[mask]
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(points):
    """Compute area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _exact_score(points):
    """Calculate the exact score: min triangle area / hull area."""
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_objective(params, beta):
    """
    Negative soft-min of triangle areas divided by hull area.
    We minimize this to maximize the score.
    """
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    # Soft min approximation
    # To prevent overflow/underflow in exp
    scaled_areas = -beta * areas
    max_val = np.max(scaled_areas)
    if max_val == -np.inf:
        return 1e10
    log_sum_exp = max_val + np.log(np.sum(np.exp(scaled_areas - max_val)))
    soft_min = -log_sum_exp / beta
    
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
        
    return -soft_min / hull_area

def construct():
    n = N
    t0 = time.time()
    deadline = 145.0 # Safety margin from 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Define initial configurations
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_reg13 = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # 2. Jittered Regular 13-gon
    rng = np.random.RandomState(42)
    pts_jit = pts_reg13 + 0.05 * rng.normal(0, 1, (n, 2))
    
    initializations = [pts_reg13, pts_jit]
    
    for init_pts in initializations:
        if time.time() - t0 > deadline - 20: # Need 20s for one full cycle
            break
            
        current_best_pts = init_pts
        current_best_score = _exact_score(init_pts)
        
        # Stage 1: L-BFGS-B with soft objective
        beta = 30.0
        try:
            x0 = init_pts.flatten()
            res1 = minimize(_soft_objective, x0, args=(beta,), method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            cand = res1.x.reshape(-1, 2)
            score1 = _exact_score(cand)
            if score1 > current_best_score:
                current_best_score = score1
                current_best_pts = cand
        except Exception:
            pass
        
        # Stage 2: Nelder-Mead on exact objective
        if time.time() - t0 < deadline - 5:
            try:
                x0 = current_best_pts.flatten()
                res2 = minimize(lambda x: -_exact_score(x.reshape(-1, 2)), x0, method='Nelder-Mead',
                                options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10, 'adaptive': True})
                cand = res2.x.reshape(-1, 2)
                score2 = _exact_score(cand)
                if score2 > current_best_score:
                    current_best_score = score2
                    current_best_pts = cand
            except Exception:
                pass
        
        if current_best_score > best_score:
            best_score = current_best_score
            best_points = current_best_pts
            
    if best_points is None:
        return initializations[0]
        
    return best_points
