"""Centrally-symmetric parameterization + DE global search + local refinement."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
H = N // 2  # 7

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            if j + 1 >= N:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
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
    """Minimum triangle area divided by convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _points_from_sym(params):
    """
    Build 14 points from 14 parameters (7 pairs, centrally symmetric).
    params shape (14,): [x0,y0, x1,y1, ..., x6,y6]
    Point i and point i+7 are negatives of each other.
    We place the 7 points in the upper half-plane (y >= 0) to ensure
    the polygon is convex when ordered properly.
    
    Actually, for central symmetry, we just need 7 points p0..p6 and
    their negatives -p0..-p6. The convex hull of all 14 points is
    centrally symmetric.
    
    To ensure convexity, we place points in angular order around origin.
    """
    # params: 7 (x,y) pairs
    pts_half = params.reshape(H, 2)
    pts_full = np.vstack([pts_half, -pts_half])
    return pts_full

def _objective(params):
    """
    Objective for optimization: minimize negative score.
    params: 14 values (7 pairs).
    """
    points = _points_from_sym(params)
    s = _score(points)
    if s < 1e-12:
        return 1e10
    return -s

def _objective_soft(params, penalty_weight=100.0):
    """
    Soft objective that penalizes non-convex configurations.
    We check if the 14 points form a convex polygon (all cross products
    same sign when ordered by angle).
    """
    points = _points_from_sym(params)
    
    # Sort points by angle around centroid
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    order = np.argsort(angles)
    sorted_pts = points[order]
    
    # Check convexity: all cross products of consecutive edges should have same sign
    n = len(sorted_pts)
    cross_sums = []
    for i in range(n):
        p0 = sorted_pts[i]
        p1 = sorted_pts[(i+1) % n]
        p2 = sorted_pts[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        cross_sums.append(cross)
    
    cross_sums = np.array(cross_sums)
    
    # If all same sign (positive or all negative), it's convex
    if np.all(cross_sums > -1e-10) or np.all(cross_sums < 1e-10):
        s = _score(points)
        if s < 1e-12:
            return 1e10
        return -s
    
    # Penalize non-convex: return large value based on how non-convex
    min_cross = np.min(np.abs(cross_sums))
    s = _score(points)
    penalty = penalty_weight * max(0, -min(cross_sums) if np.any(cross_sums < 0) else 0)
    return -s + penalty

def _generate_initial_configs(n_configs=20):
    """Generate diverse initial configurations in the symmetric parameter space."""
    configs = []
    
    # Config 1: Regular 14-gon (centrally symmetric)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    # Extract first 7 points as params
    configs.append(regular[:H].flatten().copy())
    
    # Config 2-5: Perturbed regular with different angular modes
    for k in range(1, 5):
        base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        perturbation = 0.1 * np.sin(k * 2 * np.pi * np.arange(N) / N)
        angles = base_angles + perturbation
        angles = np.mod(angles, 2*np.pi)
        angles.sort()
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        # Make centrally symmetric: average with negation
        sym_pts = np.zeros((N, 2))
        for i in range(H):
            sym_pts[i] = pts[i]
            sym_pts[i+H] = -pts[i]
        configs.append(sym_pts[:H].flatten().copy())
    
    # Config 6-10: Random points in upper half-plane, then symmetrize
    for i in range(5):
        rng = np.random.RandomState(42 + i)
        # Random angles in [0, pi) for upper half
        r_angles = rng.uniform(0, np.pi, H)
        r_angles.sort()
        # Random radii
        r_radii = rng.uniform(0.5, 1.5, H)
        pts_half = np.column_stack([r_radii * np.cos(r_angles), r_radii * np.sin(r_angles)])
        configs.append(pts_half.flatten().copy())
    
    # Config 11-15: Elliptic regular (stretch x-axis)
    for stretch in [0.5, 0.7, 1.0, 1.3, 1.5]:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([stretch * np.cos(angles), np.sin(angles)])
        # Make centrally symmetric
        sym_pts = np.zeros((N, 2))
        for i in range(H):
            sym_pts[i] = pts[i]
            sym_pts[i+H] = -pts[i]
        configs.append(sym_pts[:H].flatten().copy())
    
    # Config 16-20: Random with some structure
    for i in range(5):
        rng = np.random.RandomState(100 + i)
        # Start with regular and add random noise
        base_angles = np.linspace(0, np.pi, H, endpoint=False)
        noise = rng.normal(0, 0.15, H)
        angles = base_angles + noise
        angles.sort()
        radii = rng.uniform(0.8, 1.2, H)
        pts_half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(pts_half.flatten().copy())
    
    return configs

def _refine_local(params, time_limit):
    """Refine a configuration using Nelder-Mead in the 14D symmetric space."""
    start_time = time.time()
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e10
        return _objective(p)
    
    try:
        res = minimize(
            obj,
            params,
            method='Nelder-Mead',
            options={
                'maxiter': 200,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        return res.x
    except Exception:
        return params

def construct():
    """
    Generate 14 points to maximize min triangle area / hull area.
    
    Strategy:
    1. Use centrally symmetric parameterization (14D instead of 28D).
    2. Differential evolution for global search.
    3. Multi-start local refinement from diverse initial configs.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Define bounds for the 14 parameters (7 x-coords, 7 y-coords)
    # Points should be in a reasonable range; hull area normalization handles scale
    bounds = [(-3, 3)] * (2 * H)
    
    best_score = -1e10
    best_params = None
    best_points = None
    
    # Phase 1: Differential Evolution (global search in 14D symmetric space)
    # Use a smaller population and maxiter to fit in time budget
    try:
        de_result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.8,
            polish=True,
            init='sobol',
            updating='immediate',
            callback=lambda x, convergence: None
        )
        
        if de_result.success or de_result.fun < 1e9:
            s = -de_result.fun
            if s > best_score:
                best_score = s
                best_params = de_result.x.copy()
                best_points = _points_from_sym(de_result.x)
        
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time for refinement; return best from DE
        if best_points is not None:
            try:
                hull = ConvexHull(best_points)
                hull_area = hull.volume
                if hull_area > 1e-10:
                    scale = np.sqrt(1.0 / hull_area)
                    return best_points * scale
            except Exception:
                pass
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
            scale = np.sqrt(1.0 / hull_area)
            return pts * scale
        except Exception:
            return pts
    
    # Phase 2: Multi-start local refinement from diverse initial configs
    initial_configs = _generate_initial_configs(20)
    
    # Add DE result as one of the configs
    if best_params is not None:
        initial_configs.append(best_params.copy())
    
    # Sort configs by initial score (best first)
    config_scores = []
    for cfg in initial_configs:
        s = -_objective(cfg)
        config_scores.append((s, cfg))
    config_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top configs with local optimization
    n_to_refine = min(10, len(config_scores))
    for i in range(n_to_refine):
        if time.time() > deadline - 10:
            break
        
        s_init, cfg = config_scores[i]
        time_left = deadline - time.time() - 5
        if time_left < 3:
            break
        
        refined = _refine_local(cfg, time_left)
        s_ref = -_objective(refined)
        
        if s_ref > best_score:
            best_score = s_ref
            best_params = refined.copy()
            best_points = _points_from_sym(refined)
    
    # Phase 3: One more round of DE with tighter bounds around best if time permits
    if best_params is not None and time.time() < deadline - 40:
        try:
            tight_bounds = []
            for i in range(len(best_params)):
                lo = best_params[i] - 1.0
                hi = best_params[i] + 1.0
                tight_bounds.append((lo, hi))
            
            de_result2 = differential_evolution(
                _objective,
                tight_bounds,
                seed=123,
                maxiter=20,
                popsize=12,
                tol=1e-7,
                mutation=(0.2, 0.8),
                recombination=0.7,
                polish=True,
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result2.fun < 1e9:
                s = -de_result2.fun
                if s > best_score:
                    best_score = s
                    best_params = de_result2.x.copy()
                    best_points = _points_from_sym(de_result2.x)
        except Exception:
            pass
    
    # Final: Normalize to unit area
    if best_points is not None:
        try:
            hull = ConvexHull(best_points)
            hull_area = hull.volume
            if hull_area > 1e-10:
                scale = np.sqrt(1.0 / hull_area)
                return best_points * scale
        except Exception:
            pass
    
    # Fallback: regular 14-gon normalized
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        return pts * scale
    except Exception:
        return pts

# EVOLVE-BLOCK-END
