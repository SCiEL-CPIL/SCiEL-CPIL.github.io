"""
Strategy: Centrally symmetric polygon optimization with multi-start local refinement.
Uses diverse initial configurations and aggressive Nelder-Mead/Powell polish.
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

def _all_triangle_areas(points):
    """
    Compute all triangle areas (vectorized).
    Returns array of shape (C(n,3),) with all triangle areas.
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

def _objective_soft(params, beta=80.0):
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
    if len(areas) == 0:
        return 1e10
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # Shift for numerical stability
    min_a = np.min(areas)
    shifted = areas - min_a
    soft_min_val = -1.0/beta * np.log(np.sum(np.exp(-beta * shifted))) + min_a
    
    score = soft_min_val / hull_area
    return -score

def _generate_initial_configs():
    """
    Generate diverse initial configurations for multi-start optimization.
    Returns list of parameter arrays (shape 14,).
    """
    configs = []
    
    # 1. Regular 14-gon
    init_angles = np.linspace(0, np.pi, M, endpoint=False)
    init_radii = np.ones(M)
    init_vecs = np.column_stack((init_radii * np.cos(init_angles), init_radii * np.sin(init_angles)))
    configs.append(init_vecs.flatten())
    
    # 2. Slightly perturbed regular (random small perturbations)
    for trial in range(5):
        angles = init_angles + np.random.uniform(-0.05, 0.05, M)
        radii = 1.0 + np.random.uniform(-0.1, 0.1, M)
        vecs = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        configs.append(vecs.flatten())
    
    # 3. Two concentric rings pattern
    for r_inner in [0.5, 0.6, 0.7]:
        # 4 inner points, 3 outer points (or similar split)
        inner_angles = np.linspace(0, np.pi, 4, endpoint=False)
        outer_angles = np.linspace(0, np.pi, 3, endpoint=False)
        inner_vecs = np.column_stack((r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)))
        outer_vecs = np.column_stack((1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)))
        vecs = np.vstack((inner_vecs, outer_vecs))
        configs.append(vecs.flatten())
    
    # 4. Known Heilbronn-like pattern: alternating radii
    for phase in [0, 0.3, 0.6]:
        angles = init_angles + phase
        radii = np.array([1.0, 0.85, 1.0, 0.85, 1.0, 0.85, 1.0])
        vecs = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        configs.append(vecs.flatten())
    
    # 5. Random but bounded
    for _ in range(3):
        vecs = np.random.uniform(-1.2, 1.2, (M, 2))
        # Ensure no zero vectors
        norms = np.linalg.norm(vecs, axis=1)
        vecs[norms < 0.1] = 0.5
        configs.append(vecs.flatten())
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator kills at 150s)
    
    # Generate diverse initial configurations
    init_configs = _generate_initial_configs()
    
    best_params = init_configs[0].copy()
    best_score_val = -_objective(best_params)
    
    # Phase 1: Multi-start Nelder-Mead with hard objective
    # Budget: ~80s for this phase
    time_per_start = 80.0 / len(init_configs)
    
    for i, init in enumerate(init_configs):
        if time.time() > deadline - 50:
            break
        
        try:
            nm_res = minimize(
                _objective,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 1500,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            if nm_res.fun < _objective(best_params):
                best_params = nm_res.x
                best_score_val = -nm_res.fun
        except Exception:
            pass
    
    # Phase 2: Powell's method for more aggressive local search
    if time.time() < deadline - 30:
        try:
            pow_res = minimize(
                _objective,
                best_params,
                method='Powell',
                options={
                    'maxiter': 1000,
                    'xtol': 1e-10,
                    'ftol': 1e-12,
                    'adaptive': True
                }
            )
            if pow_res.fun < _objective(best_params):
                best_params = pow_res.x
                best_score_val = -pow_res.fun
        except Exception:
            pass
    
    # Phase 3: Soft objective polish with multiple beta values
    if time.time() < deadline - 15:
        for beta in [100.0, 150.0, 200.0]:
            if time.time() > deadline - 10:
                break
            try:
                nm_soft_res = minimize(
                    lambda p: _objective_soft(p, beta=beta),
                    best_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 800,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                # Check if the soft result is actually better with the hard metric
                hard_score_soft_params = -_objective(nm_soft_res.x)
                if hard_score_soft_params > best_score_val:
                    best_params = nm_soft_res.x
                    best_score_val = hard_score_soft_params
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            final_res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if final_res.fun < _objective(best_params):
                best_params = final_res.x
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
