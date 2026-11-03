"""
Strategy: Centrally symmetric polygon optimization with multi-start local refinement.
Exploits the symmetry of optimal Heilbronn configurations for even N.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
M = N // 2  # 7 pairs of antipodal points

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized implementation for speed.
    """
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            # k ranges from j+1 to n-1
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _all_triangle_areas(points):
    """
    Compute all triangle areas as a flat array.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            a = 0.5 * np.abs(cross)
            areas.extend(a)
    return np.array(areas)

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _points_from_sym_params(params):
    """
    Reconstruct points from 7 independent 2D vectors (centrally symmetric).
    params: array of shape (14,) -> 7 vectors (x1, y1, ..., x7, y7)
    The points are v1, v2, ..., v7, -v1, -v2, ..., -v7.
    """
    vecs = params.reshape(M, 2)
    # To ensure convexity and avoid self-intersection or degenerate hulls,
    # we should ideally sort by angle, but for optimization we can just return
    # the points and let the score function penalize bad configurations.
    # However, for ConvexHull to work correctly and represent the intended polygon,
    # the points should be in order. 
    # A robust way: compute angles and sort.
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    # Sort the vectors by angle
    sorted_indices = np.argsort(angles)
    sorted_vecs = vecs[sorted_indices]
    
    # Create full set of points: sorted_vecs and -sorted_vecs (reversed to maintain convex order?)
    # Actually, if we take sorted_vecs and append -sorted_vecs[::-1], we get a convex polygon
    # if the vectors are in angular order around the origin.
    # Let's verify: v1 (angle a1), v2 (a2)... v7 (a7).
    # -v1 has angle a1 + pi.
    # If we list v1..v7, then -v7..-v1, the angles are a1..a7, a7+pi..a1+pi.
    # This forms a closed loop around the origin.
    pts_pos = sorted_vecs
    pts_neg = -sorted_vecs[::-1]
    points = np.vstack((pts_pos, pts_neg))
    
    return points

def _objective(params):
    """
    Objective function to minimize (negative score).
    Uses a soft minimum to improve differentiability for local optimizers.
    """
    points = _points_from_sym_params(params)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def _objective_soft(params):
    """
    Objective using a soft-min for the triangle areas to smooth the landscape.
    """
    points = _points_from_sym_params(params)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10
        
    areas = _all_triangle_areas(points)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    beta = 50.0
    # Shift for numerical stability
    shifted = areas - np.min(areas)
    soft_min_val = -1.0/beta * np.log(np.sum(np.exp(-beta * shifted))) + np.min(areas)
    
    score = soft_min_val / hull_area
    return -score

def _generate_initial_configs():
    """
    Generate diverse initial configurations for multi-start optimization.
    Returns a list of parameter arrays.
    """
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, np.pi, M, endpoint=False)
    radii = np.ones(M)
    vecs = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    configs.append(vecs.flatten())
    
    # 2-10: Perturbed regular polygons with varying radius and angle noise
    for i in range(9):
        rng = np.random.default_rng(42 + i)
        base_angles = np.linspace(0, np.pi, M, endpoint=False)
        angle_noise = rng.normal(0, 0.15, M)
        angles_perturbed = base_angles + angle_noise
        # Ensure angles are within [0, pi) for the first half
        angles_perturbed = angles_perturbed % np.pi
        
        base_radii = np.ones(M)
        radius_noise = rng.normal(0, 0.1, M)
        radii_perturbed = np.clip(base_radii + radius_noise, 0.5, 1.5)
        
        vecs = np.column_stack((radii_perturbed * np.cos(angles_perturbed), 
                                radii_perturbed * np.sin(angles_perturbed)))
        configs.append(vecs.flatten())
    
    # 11-14: More structured variations
    # Elliptical-like
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.3, 0.7), (0.7, 1.3)]:
        angles = np.linspace(0, np.pi, M, endpoint=False)
        vecs = np.column_stack((rx * np.cos(angles), ry * np.sin(angles)))
        configs.append(vecs.flatten())
        
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    # Generate diverse initial configurations
    initial_configs = _generate_initial_configs()
    
    best_params = initial_configs[0].copy()
    best_score_val = -_objective(best_params)
    
    # Multi-start Nelder-Mead with time budgeting
    num_starts = len(initial_configs)
    time_per_start = (deadline - start_time - 15) / num_starts  # Reserve 15s for final polish
    
    for idx, init_params in enumerate(initial_configs):
        if time.time() > deadline - 15:
            break
            
        # Local refinement with Nelder-Mead
        try:
            nm_res = minimize(
                _objective,
                init_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if nm_res.fun < _objective(best_params):
                best_params = nm_res.x.copy()
                best_score_val = -nm_res.fun
        except Exception:
            pass
        
        # Soft objective refinement for the best candidate so far
        if time.time() < deadline - 20:
            try:
                nm_soft_res = minimize(
                    _objective_soft,
                    best_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                hard_score = -_objective(nm_soft_res.x)
                if hard_score > best_score_val:
                    best_params = nm_soft_res.x.copy()
                    best_score_val = hard_score
            except Exception:
                pass
    
    # Final high-precision polish on the best configuration
    if time.time() < deadline - 5:
        try:
            final_res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if final_res.fun < _objective(best_params):
                best_params = final_res.x.copy()
                best_score_val = -final_res.fun
        except Exception:
            pass

    # Reconstruct final points
    points = _points_from_sym_params(best_params)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        # Fallback to regular polygon if hull fails
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        return final_points
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = points * scale
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
