"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

_start_time = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

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
    """Compute convex hull area using scipy."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume = area
    except Exception:
        return 1.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _all_triangle_areas(pts)
    min_ta = np.min(areas)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    return min_ta / hull_a

def _soft_min_objective(params, beta, m, k):
    """
    Soft-min approximation of the min triangle area objective.
    params: [r_inner, phase, *radial_perturbations]
    We use a log-sum-exp based soft minimum to make the objective smooth.
    """
    r_inner = params[0]
    phase = params[1]
    
    if r_inner <= 0.01 or r_inner >= 0.99:
        return 1e10
    
    # Generate base multi-shell points
    outer_angles = 2 * np.pi * np.arange(m) / m
    inner_angles = 2 * np.pi * np.arange(k) / k + phase
    
    outer_pts = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    inner_pts = np.column_stack((r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)))
    
    base_pts = np.vstack([outer_pts, inner_pts])
    
    # Apply radial perturbations if provided
    n_params = len(params)
    if n_params > 2:
        radial_perts = params[2:]
        for i, rp in enumerate(radial_perts):
            if i < len(base_pts):
                angle = np.arctan2(base_pts[i, 1], base_pts[i, 0])
                r = np.sqrt(base_pts[i, 0]**2 + base_pts[i, 1]**2)
                r_new = r * (1.0 + rp * 0.1)  # Small radial perturbation
                base_pts[i, 0] = r_new * np.cos(angle)
                base_pts[i, 1] = r_new * np.sin(angle)
    
    areas = _all_triangle_areas(base_pts)
    if len(areas) == 0:
        return 1e10
    
    # Soft minimum using log-sum-exp trick
    # soft_min(x) ≈ -(1/beta) * log(sum(exp(-beta * x_i)))
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return 1e10
    soft_min_val = -(1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    hull_a = _hull_area(base_pts)
    if hull_a < 1e-12:
        return 1e10
    
    return soft_min_val / hull_a

def _generate_multi_shell(m, k, r_inner, phase):
    """Generate points: m points on outer unit circle, k points on inner circle of radius r_inner, rotated by phase."""
    outer_angles = 2 * np.pi * np.arange(m) / m
    inner_angles = 2 * np.pi * np.arange(k) / k + phase
    
    outer_pts = np.column_stack((np.cos(outer_angles), np.sin(outer_angles)))
    inner_pts = np.column_stack((r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)))
    
    return np.vstack([outer_pts, inner_pts])

def _neg_score_multi_shell(params, m, k):
    """Negative score for minimization. params = [r_inner, phase]."""
    r_inner = params[0]
    phase = params[1]
    if r_inner <= 0 or r_inner >= 1:
        return 1e10
    pts = _generate_multi_shell(m, k, r_inner, phase)
    return -_score(pts)

def _refine_with_l_bfgsb(pts, time_budget=30.0):
    """Refine individual point coordinates using L-BFGS-B with soft-min objective."""
    n = len(pts)
    
    def _neg_soft_min(x, beta):
        p = x.reshape(-1, 2)
        # Ensure points stay in reasonable bounds
        if np.any(np.abs(p) > 5):
            return 1e10
        areas = _all_triangle_areas(p)
        if len(areas) == 0:
            return 1e10
        scaled = -beta * areas
        max_scaled = np.max(scaled)
        if max_scaled > 700:
            return 1e10
        soft_min_val = -(1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
        hull_a = _hull_area(p)
        if hull_a < 1e-12:
            return 1e10
        return soft_min_val / hull_a
    
    x0 = pts.flatten()
    bounds = [(-3, 3)] * (2 * n)
    
    # Start with moderate beta and increase
    best_x = x0.copy()
    best_val = _neg_soft_min(x0, 50.0)
    
    for beta in [50.0, 100.0, 200.0, 500.0]:
        if time.time() - _start_time > _TIME_LIMIT - 10:
            break
        try:
            res = minimize(_neg_soft_min, best_x, args=(beta,), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(-1, 2)

def construct():
    global _start_time
    elapsed = time.time() - _start_time
    remaining = _TIME_LIMIT - elapsed
    if remaining < 5:
        # Not enough time, return regular 13-gon
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_pts = None
    best_score = -1.0
    
    # Try different partitions (m, k) with m + k = 13
    partitions = [(7, 6), (8, 5), (9, 4), (10, 3), (11, 2), (12, 1)]
    
    for m, k in partitions:
        if time.time() - _start_time > _TIME_LIMIT - 25:
            break
        
        # Optimize (r_inner, phase) with Nelder-Mead
        best_params = None
        best_val = 1e10
        
        # Try multiple starting points for robustness
        for r0 in [0.3, 0.5, 0.7, 0.4, 0.6]:
            for ph0 in [0.0, np.pi / max(k, 1), np.pi / max(m, 1), np.pi / (2 * max(k, 1))]:
                x0 = [r0, ph0]
                try:
                    res = minimize(_neg_score_multi_shell, x0, args=(m, k),
                                   method='Nelder-Mead',
                                   options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
                    if res.fun < best_val:
                        best_val = res.fun
                        best_params = res.x
                except Exception:
                    pass
        
        if best_params is not None:
            pts = _generate_multi_shell(m, k, best_params[0], best_params[1])
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback to regular 13-gon
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_pts)
    
    # Refine individual points with L-BFGS-B if time allows
    if time.time() - _start_time < _TIME_LIMIT - 35:
        try:
            refined = _refine_with_l_bfgsb(best_pts, time_budget=30.0)
            refined_score = _score(refined)
            if refined_score > best_score:
                best_pts = refined
                best_score = refined_score
        except Exception:
            pass
    
    # Final check
    if time.time() - _start_time > _TIME_LIMIT - 5:
        pass  # Just return what we have
    
    return best_pts
# EVOLVE-BLOCK-END
