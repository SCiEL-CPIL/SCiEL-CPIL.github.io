"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area if min_area != float('inf') else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha=50.0):
    """
    Smooth approximation of the minimum triangle area.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    log_sum = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                min_a = np.min(areas)
                if min_a > 0:
                    shifted = -alpha * (areas - min_a)
                    log_sum_partial = np.log(np.sum(np.exp(shifted))) - alpha * min_a
                    if log_sum_partial > log_sum:
                        log_sum = log_sum_partial
    if log_sum == -np.inf:
        return 0.0
    return -log_sum / alpha

def _hull_area_safe(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1.0

def _smooth_score(pts):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area_safe(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _soft_min_triangle_area(pts, alpha=50.0) / hull_a

def _neg_smooth_score(params):
    pts = params.reshape(-1, 2)
    return -_smooth_score(pts)

def _construct_ring_config(params):
    """
    Construct 13 points: 1 center + 12 points on a ring with fixed angular spacing.
    params: array of 13 values
      - params[0]: center x
      - params[1]: center y  
      - params[2:14]: 12 radial perturbations (relative to base radius 1.0)
    
    The 12 outer points are at angles 0, 2pi/12, 4pi/12, ..., 22pi/12
    with radii 1.0 + params[2:14]
    """
    cx, cy = params[0], params[1]
    base_r = 1.0
    radii = base_r + params[2:14]
    # Ensure positive radii
    radii = np.maximum(radii, 0.1)
    
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    outer_x = radii * np.cos(angles)
    outer_y = radii * np.sin(angles)
    
    center = np.array([[cx, cy]])
    outer = np.column_stack([outer_x, outer_y])
    return np.vstack([center, outer])

def _neg_ring_score(params):
    pts = _construct_ring_config(params)
    return -_score(pts)

def _neg_ring_smooth_score(params):
    pts = _construct_ring_config(params)
    return -_smooth_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    
    # Strategy 1: Differential Evolution on the 13D symmetric ring parameter space
    # params: [cx, cy, r0, r1, ..., r11] where ri are radial perturbations
    # Bounds: center in [-0.5, 0.5], radii in [0.3, 1.7]
    bounds = [(-0.5, 0.5), (-0.5, 0.5)] + [(-0.7, 0.7)] * 12
    
    # First, try a quick DE run to find a good starting point
    try:
        res_de = differential_evolution(
            _neg_ring_score,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=True
        )
        if time.time() < deadline - 30.0:
            de_params = res_de.x
            de_pts = _construct_ring_config(de_params)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
    except Exception:
        pass
    
    # Strategy 2: Refine the best ring configuration using Nelder-Mead on the 13D space
    if time.time() < deadline - 20.0:
        # Start from the best DE result or a regular hexagon-like init
        if best_score > 0:
            # Invert the best_pts to get approximate params (rough)
            # For simplicity, just use the DE result
            try:
                res_nm = minimize(
                    _neg_ring_score,
                    res_de.x,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                nm_pts = _construct_ring_config(res_nm.x)
                nm_score = _score(nm_pts)
                if nm_score > best_score:
                    best_score = nm_score
                    best_pts = nm_pts.copy()
            except Exception:
                pass
        else:
            # Fallback: regular 12-gon + center
            angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            outer = np.column_stack([np.cos(angles), np.sin(angles)])
            center = np.array([[0.0, 0.0]])
            init_pts = np.vstack([center, outer])
            init_score = _score(init_pts)
            if init_score > best_score:
                best_score = init_score
                best_pts = init_pts.copy()
    
    # Strategy 3: Free-space optimization from the best ring configuration
    # This allows breaking the ring symmetry if beneficial
    if time.time() < deadline - 15.0:
        current_pts = best_pts.copy()
        current_score = _score(current_pts)
        
        # Nelder-Mead on full 26D space
        try:
            res_free = minimize(
                _neg_score,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            free_pts = res_free.x.reshape(-1, 2)
            free_score = _score(free_pts)
            if free_score > best_score:
                best_score = free_score
                best_pts = free_pts.copy()
        except Exception:
            pass
    
    # Strategy 4: Final polish
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            final_pts = res_final.x.reshape(-1, 2)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts.copy()
        except Exception:
            pass

    return best_pts

# EVOLVE-BLOCK-END
