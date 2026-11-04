"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 13

# Precompute angles for the fixed angular grid
# Uniformly spaced angles: theta_i = 2*pi*i/N
ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
COS_A = np.cos(ANGLES)
SIN_A = np.sin(ANGLES)

# Precompute triangle indices
from itertools import combinations
TRI_INDICES = list(combinations(range(N), 3))
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _points_from_radial(r):
    """Convert radial distances to 2D points using fixed uniform angles."""
    # r is (N,)
    x = r * COS_A
    y = r * SIN_A
    return np.column_stack([x, y])

def _tri_areas_from_radial(r):
    """Compute all triangle areas given radial distances."""
    pts = _points_from_radial(r)
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area_from_radial(r):
    """Compute convex hull area given radial distances."""
    pts = _points_from_radial(r)
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_radial(r):
    """Score = min triangle area / convex hull area for radial configuration."""
    ha = _hull_area_from_radial(r)
    if ha < 1e-12:
        return 0.0
    min_area = np.min(_tri_areas_from_radial(r))
    return min_area / ha

def _soft_min_ratio_radial(r, temperature):
    """
    Smooth surrogate for optimization: soft-min of triangle areas / hull area.
    """
    ha = _hull_area_from_radial(r)
    if ha < 1e-12:
        return -1e9
    
    areas = _tri_areas_from_radial(r)
    areas = np.maximum(areas, 1e-15)
    
    # soft_min = -T * log(sum(exp(-a/T)))
    scaled = -areas / temperature
    scaled_max = np.max(scaled)
    log_sum = np.log(np.sum(np.exp(scaled - scaled_max))) + scaled_max
    soft_min_area = -log_sum * temperature
    
    return soft_min_area / ha

def _optimize_radial_restart(r0, time_budget, rng):
    """
    Optimize a single radial configuration starting from r0.
    """
    start_time = time.time()
    best_r = r0.copy()
    best_score = _score_radial(best_r)
    
    # Bounds for radii: [0.1, 1.5] to allow some shape distortion but keep compact
    bounds = [(0.1, 1.5)] * N
    
    # Stage 1: Coarse optimization with higher temperature
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
    
    for temp in temperatures:
        if time.time() - start_time > time_budget - 2:
            break
            
        def obj(x):
            val = _soft_min_ratio_radial(x, temp)
            if val < -1e8:
                return 1e8
            return -val
        
        try:
            res = minimize(
                obj, best_r, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            cand_r = res.x
            cand_score = _score_radial(cand_r)
            if cand_score > best_score:
                best_score = cand_score
                best_r = cand_r.copy()
        except Exception:
            pass
            
    # Stage 2: Fine polish with very low temperature
    if time.time() - start_time < time_budget - 1:
        temp_final = 0.0005
        def obj_final(x):
            val = _soft_min_ratio_radial(x, temp_final)
            if val < -1e8:
                return 1e8
            return -val
        
        try:
            res = minimize(
                obj_final, best_r, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            cand_r = res.x
            cand_score = _score_radial(cand_r)
            if cand_score > best_score:
                best_score = cand_score
                best_r = cand_r.copy()
        except Exception:
            pass
            
    return best_r, best_score

def construct():
    """Main entry point: find optimal 13-point configuration using radial parameterization."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse radial starting points
    starts = []
    
    # 1. Regular polygon (all radii = 1)
    starts.append(np.ones(N))
    
    # 2. Slightly perturbed regular polygon
    for scale in [0.05, 0.1, 0.2, 0.3]:
        r_pert = np.ones(N) + rng.normal(0, scale, N)
        r_pert = np.clip(r_pert, 0.1, 1.5)
        starts.append(r_pert)
        
    # 3. Alternating radii (star-like)
    r_alt = np.ones(N)
    r_alt[::2] *= 1.2
    r_alt[1::2] *= 0.8
    starts.append(r_alt)
    
    r_alt2 = np.ones(N)
    r_alt2[::3] *= 1.3
    r_alt2[1::3] *= 0.9
    r_alt2[2::3] *= 1.1
    starts.append(r_alt2)
    
    # 4. Random radii
    for _ in range(10):
        r_rand = rng.uniform(0.5, 1.2, N)
        starts.append(r_rand)
        
    # 5. Gradient-like profiles
    r_grad = np.linspace(0.8, 1.2, N)
    starts.append(r_grad)
    r_grad2 = np.linspace(1.2, 0.8, N)
    starts.append(r_grad2)

    # Evaluate initial scores to sort
    scored_starts = []
    for r_start in starts:
        try:
            s = _score_radial(r_start)
            scored_starts.append((s, r_start))
        except Exception:
            scored_starts.append((0.0, r_start))
            
    scored_starts.sort(key=lambda x: -x[0])
    
    # Allocate time per start
    n_starts = len(scored_starts)
    base_time = (total_budget - 10.0) / n_starts
    
    for i, (init_score, r_start) in enumerate(scored_starts):
        if time.time() - start_time > total_budget - 5:
            break
            
        remaining = total_budget - (time.time() - start_time)
        # Give more time to better starts
        quality_factor = 1.0 + 0.5 * (init_score / max(best_ratio, 1e-9)) if best_ratio > 0 else 1.0
        alloc_time = min(base_time * quality_factor, remaining - 2.0)
        
        if alloc_time < 2.0:
            continue
            
        try:
            opt_r, opt_score = _optimize_radial_restart(r_start, alloc_time, rng)
            if opt_score > best_ratio:
                best_ratio = opt_score
                best_points = _points_from_radial(opt_r)
        except Exception:
            pass
            
    # If no valid points found, fallback to regular polygon
    if best_points is None:
        best_points = _points_from_radial(np.ones(N))
        
    return best_points
