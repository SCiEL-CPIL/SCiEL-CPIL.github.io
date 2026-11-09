import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _negative_score(points):
    """
    Objective function for minimizer.
    """
    return -_score(points)

def _regular_polygon(n, radius=1.0):
    """Generate regular n-gon centered at origin."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _normalize_points(points):
    """
    Normalize points to have centroid at origin and unit variance (or similar)
    to prevent drift, although score is invariant.
    This helps numerical stability of the optimizer.
    """
    # Center
    points = points - np.mean(points, axis=0)
    # Scale to have mean squared distance 1
    dist_sq = np.sum(points**2, axis=1)
    mean_dist_sq = np.mean(dist_sq)
    if mean_dist_sq > 1e-12:
        points = points / np.sqrt(mean_dist_sq)
    return points

def _cyclic_symmetric_params_to_points(params):
    """
    Convert 3 parameters (angle_offset, r_even, r_odd) to 14 points.
    r_even is radius for even indices, r_odd for odd indices.
    The angular positions are fixed at 2*pi*i/14.
    """
    angle_offset, r_even, r_odd = params
    angles = np.arange(N) * (2 * np.pi / N) + angle_offset
    radii = np.zeros(N)
    radii[0::2] = r_even
    radii[1::2] = r_odd
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _score_cyclic_params(params):
    """
    Score for cyclic symmetric parameters.
    """
    points = _cyclic_symmetric_params_to_points(params)
    return _score(points)

def _negative_score_cyclic_params(params):
    """
    Negative score for cyclic symmetric parameters (for minimization).
    """
    return -_score_cyclic_params(params)

def _general_points_to_params(points):
    """
    Convert 14 points (28 coords) to a parameter vector.
    We use the raw coordinates, but normalize them first for stability.
    """
    pts = _normalize_points(points)
    return pts.flatten()

def _params_to_general_points(params):
    """
    Convert parameter vector to 14 points.
    """
    points = params.reshape(N, 2)
    return _normalize_points(points)

def _score_general_params(params):
    """
    Score for general parameters.
    """
    points = _params_to_general_points(params)
    return _score(points)

def _negative_score_general_params(params):
    """
    Negative score for general parameters (for minimization).
    """
    return -_score_general_params(params)

def _refine_local(params, func, method='Nelder-Mead', maxiter=200, timeout=10):
    """
    Local refinement using Nelder-Mead.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    def callback(xk):
        if time.time() - start_time > timeout:
            raise KeyboardInterrupt()
    
    try:
        res = minimize(func, params, method=method,
                       options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
                       callback=callback)
        if time.time() - start_time > timeout:
            return params
        return res.x
    except KeyboardInterrupt:
        return params
    except Exception:
        return params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    def update_best(points, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = points.copy()
            best_score = score

    # Strategy 1: Cyclic Symmetric Optimization
    # Search space: 3 parameters (angle_offset, r_even, r_odd)
    # This restricts the shape to 2-fold rotational symmetry (or similar)
    # but allows for "alternating" radii which can improve min triangle area.
    
    from scipy.optimize import differential_evolution, minimize
    
    # Bounds for cyclic params:
    # angle_offset in [0, 2*pi/N] (due to symmetry, only need one sector)
    # radii in [0.5, 1.5] (normalized scale)
    
    bounds_cyclic = [(0, 2 * np.pi / N), (0.5, 1.5), (0.5, 1.5)]
    
    # Global search: Differential Evolution on cyclic space
    # DE is fast in low dimensions
    try:
        if time.time() < deadline - 60:
            de_res = differential_evolution(
                _negative_score_cyclic_params,
                bounds_cyclic,
                seed=42,
                maxiter=100,
                tol=1e-8,
                popsize=20,
                init='sobol'
            )
            if de_res.fun < -best_score:
                best_params_cyclic = de_res.x
                best_score_cyclic = -de_res.fun
                best_points_cyclic = _cyclic_symmetric_params_to_points(best_params_cyclic)
                update_best(best_points_cyclic, best_score_cyclic)
            else:
                best_params_cyclic = de_res.x
                best_score_cyclic = -de_res.fun
                best_points_cyclic = _cyclic_symmetric_params_to_points(best_params_cyclic)
                # Update best if it's better than current global best
                s = _score(best_points_cyclic)
                if s > best_score:
                    update_best(best_points_cyclic, s)
            
            # Local refinement on cyclic best
            if time.time() < deadline - 30:
                refined_params = _refine_local(best_params_cyclic, _negative_score_cyclic_params, timeout=10)
                refined_points = _cyclic_symmetric_params_to_points(refined_params)
                s = _score(refined_points)
                if s > best_score:
                    update_best(refined_points, s)
    except Exception:
        pass

    # Strategy 2: General Optimization with Random Restarts
    # Use the best cyclic result as one of the starts
    # Also use regular polygon
    # And some random convex points
    
    starts = []
    
    # Start 1: Regular polygon
    reg_poly = _regular_polygon(N)
    reg_poly_norm = _normalize_points(reg_poly)
    starts.append(reg_poly_norm.flatten())
    
    # Start 2: Best cyclic result
    if best_points is not None:
        starts.append(_normalize_points(best_points).flatten())
    else:
        # Fallback: regular polygon in cyclic params
        # Regular polygon has r_even = r_odd = 1, angle_offset = 0
        reg_cyclic = _cyclic_symmetric_params_to_points([0, 1.0, 1.0])
        starts.append(_normalize_points(reg_cyclic).flatten())
    
    # Start 3-10: Random convex-ish points
    rng = np.random.default_rng(123)
    for i in range(8):
        if time.time() > deadline - 20:
            break
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = 1.0 + rng.normal(0, 0.1, N)
        radii = np.abs(radii)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        starts.append(_normalize_points(pts).flatten())
    
    # Run Nelder-Mead on each start
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        # We have at most len(starts) - i restarts left.
        # Give each restart a proportional slice, capped at 15s
        time_per_restart = min(15.0, remaining / max(1, len(starts) - i))
        
        refined_params = _refine_local(x0, _negative_score_general_params, timeout=time_per_restart)
        refined_points = _params_to_general_points(refined_params)
        s = _score(refined_points)
        if s > best_score:
            update_best(refined_points, s)
    
    # Final refinement on the overall best
    if time.time() < deadline - 5 and best_points is not None:
        x0 = _normalize_points(best_points).flatten()
        # Use a slightly larger budget for final polish
        time_left = deadline - time.time() - 2
        if time_left > 5:
            refined_params = _refine_local(x0, _negative_score_general_params, timeout=time_left)
            refined_points = _params_to_general_points(refined_params)
            s = _score(refined_points)
            if s > best_score:
                update_best(refined_points, s)
    
    if best_points is None:
        best_points = reg_poly
        
    return best_points

# EVOLVE-BLOCK-END
