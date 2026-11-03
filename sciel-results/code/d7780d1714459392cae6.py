"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_DEADLINE = 150.0
_START = time.time()
_SAFETY = 10.0

TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _triangle_areas(pts):
    """All triangle areas, vectorized."""
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _min_triangle_area(pts):
    return np.min(_triangle_areas(pts))

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    ha = _hull_area(pts)
    if ha <= 0:
        return 0.0
    return _min_triangle_area(pts) / ha

def _soft_min(areas, temperature=0.05):
    """Soft minimum approximation for smooth optimization."""
    neg = -areas / temperature
    max_neg = np.max(neg)
    log_sum = max_neg + np.log(np.sum(np.exp(neg - max_neg)))
    return -temperature * log_sum

def _neg_soft_score(x, temperature=0.05):
    """Negative soft-min score for minimization."""
    pts = x.reshape(N, 2)
    ha = _hull_area(pts)
    if ha <= 0:
        return 1.0
    areas = _triangle_areas(pts)
    sm = _soft_min(areas, temperature)
    return -(sm / ha)

def _generate_concentric_init(n_rings=2):
    """Generate points on concentric circles."""
    pts = [np.array([0.5, 0.5])]  # center
    remaining = N - 1
    per_ring = []
    for r in range(n_rings):
        count = remaining // (n_rings - r) if r < n_rings - 1 else remaining
        per_ring.append(count)
        remaining -= count
    # Adjust last ring
    per_ring[-1] += remaining
    
    radius = 0.45
    for ring_idx, count in enumerate(per_ring):
        if count <= 0:
            continue
        r = radius * (ring_idx + 1) / n_rings
        angles = 2 * np.pi * np.arange(count) / count + ring_idx * 0.3
        ring_pts = np.column_stack([0.5 + r * np.cos(angles), 0.5 + r * np.sin(angles)])
        pts.append(ring_pts)
    
    all_pts = np.vstack(pts)
    # If we have more than N points, trim; if fewer, add random
    if len(all_pts) > N:
        all_pts = all_pts[:N]
    elif len(all_pts) < N:
        extra = np.random.rand(N - len(all_pts), 2) * 0.8 + 0.1
        all_pts = np.vstack([all_pts, extra])
    return all_pts

def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    
    # Concentric circles with different parameters
    for n_rings in [2, 3]:
        for offset in [0.0, 0.2, 0.5]:
            pts = _generate_concentric_init(n_rings)
            if offset > 0:
                angles = 2 * np.pi * np.arange(N) / N
                rot = np.array([[np.cos(offset), -np.sin(offset)],
                               [np.sin(offset), np.cos(offset)]])
                pts = (pts - 0.5) @ rot.T + 0.5
            inits.append(pts)
    
    # Regular polygon
    angles = 2 * np.pi * np.arange(N) / N
    pts_reg = np.column_stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)])
    inits.append(pts_reg)
    
    # Jittered regular polygon
    rng = np.random.RandomState(42)
    for _ in range(3):
        jitter = rng.rand(N, 2) * 0.08 - 0.04
        inits.append(pts_reg + jitter)
    
    return inits

def _targeted_worst_triangle_opt(pts, max_iter=200):
    """
    Iteratively find the worst (minimum area) triangle and optimize its three vertices.
    This is a structural change: instead of global search, we focus on the bottleneck.
    """
    pts = pts.copy()
    best_score = _score(pts)
    
    for iteration in range(max_iter):
        if time.time() - _START > _DEADLINE - _SAFETY - 5.0:
            break
        
        areas = _triangle_areas(pts)
        worst_idx = np.argmin(areas)
        i, j, k = TRI_INDICES[worst_idx]
        
        # Optimize points i, j, k to maximize the minimum area
        # We use a local optimization on these 3 points (6 variables)
        def _obj(x6):
            new_pts = pts.copy()
            new_pts[i] = x6[0:2]
            new_pts[j] = x6[2:4]
            new_pts[k] = x6[4:6]
            ha = _hull_area(new_pts)
            if ha <= 0:
                return 1.0
            return -_min_triangle_area(new_pts) / ha
        
        x0 = np.concatenate([pts[i], pts[j], pts[k]])
        bounds = [(0, 1)] * 6
        
        try:
            res = minimize(_obj, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 100, 'ftol': 1e-10})
            new_pts = pts.copy()
            new_pts[i] = res.x[0:2]
            new_pts[j] = res.x[2:4]
            new_pts[k] = res.x[4:6]
            
            new_score = _score(new_pts)
            if new_score > best_score:
                pts = new_pts
                best_score = new_score
        except Exception:
            pass
    
    return pts, best_score

def _refine_soft_min(pts, temperature=0.05, max_iter=300):
    """Refine using soft-min objective with L-BFGS-B."""
    x0 = pts.flatten()
    bounds = [(0, 1)] * (2 * N)
    
    try:
        res = minimize(lambda x: _neg_soft_score(x, temperature), x0,
                      method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': max_iter, 'ftol': 1e-10})
        return res.x.reshape(N, 2), -_neg_soft_score(res.x, temperature)
    except Exception:
        return pts, _score(pts)

def _refine_nelder_mead(pts, max_iter=500):
    """Refine using Nelder-Mead on the exact score."""
    x0 = pts.flatten()
    try:
        res = minimize(lambda x: -_score(x.reshape(N, 2)), x0,
                      method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        return res.x.reshape(N, 2), _score(res.x.reshape(N, 2))
    except Exception:
        return pts, _score(pts)

def construct():
    global _START, _DEADLINE
    
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    
    if remaining < 10.0:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)])
    
    inits = _generate_initializations()
    
    best_pts = None
    best_score = -np.inf
    
    # Evaluate all initializations
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    # Phase 1: Targeted worst-triangle optimization from best init
    if time.time() - _START < _DEADLINE - _SAFETY - 20.0:
        pts_t, score_t = _targeted_worst_triangle_opt(best_pts, max_iter=100)
        if score_t > best_score:
            best_score = score_t
            best_pts = pts_t
    
    # Phase 2: Soft-min refinement
    if time.time() - _START < _DEADLINE - _SAFETY - 10.0:
        pts_s, score_s = _refine_soft_min(best_pts, temperature=0.03, max_iter=200)
        if score_s > best_score:
            best_score = score_s
            best_pts = pts_s
    
    # Phase 3: Nelder-Mead polish
    if time.time() - _START < _DEADLINE - _SAFETY - 5.0:
        pts_nm, score_nm = _refine_nelder_mead(best_pts, max_iter=300)
        if score_nm > best_score:
            best_score = score_nm
            best_pts = pts_nm
    
    # Phase 4: Another round of targeted optimization
    if time.time() - _START < _DEADLINE - _SAFETY - 3.0:
        pts_t2, score_t2 = _targeted_worst_triangle_opt(best_pts, max_iter=50)
        if score_t2 > best_score:
            best_score = score_t2
            best_pts = pts_t2
    
    # Final Nelder-Mead polish
    if time.time() - _START < _DEADLINE - _SAFETY - 2.0:
        pts_final, score_final = _refine_nelder_mead(best_pts, max_iter=200)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    return best_pts
# EVOLVE-BLOCK-END
