import numpy as np

N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _all_triangle_areas(points):
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
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
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        return 1.0

def _score(points):
    areas = _all_triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_min_objective(params, beta=50.0):
    """Negative smooth min area / hull area."""
    points = params.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    # Soft min: -logsumexp(-beta * areas) / beta
    neg_beta = -beta * areas
    max_val = np.max(neg_beta)
    lse = max_val + np.log(np.sum(np.exp(neg_beta - max_val)))
    soft_min = -lse / beta
    hull_area = _hull_area(points)
    if hull_area <= 0:
        return 1e10
    return -soft_min / hull_area

def construct():
    n = N
    t0 = time.time()
    
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points_init = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = points_init.copy()
    best_score = _score(best_points)
    
    # Local optimization using soft-min surrogate
    try:
        x0 = points_init.flatten()
        result = minimize(_soft_min_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
        
        if time.time() - t0 < 100:
            candidate = result.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
    except Exception:
        pass
    
    # Additional refinement with smaller steps if time permits
    if time.time() - t0 < 90:
        try:
            x0 = best_points.flatten()
            result2 = minimize(_soft_min_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True})
            candidate = result2.x.reshape(-1, 2)
            cand_score = _score(candidate)
            if cand_score > best_score:
                best_score = cand_score
                best_points = candidate.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
