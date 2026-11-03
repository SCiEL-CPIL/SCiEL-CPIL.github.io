"""
Full 28D parameterization + Multi-start Nelder-Mead with coarse-to-fine refinement.
Strategy:
1. Represent 14 points as 28 coordinates, but constrain centroid to (0,0) and fix scale (e.g., sum of squared distances = 1) to reduce effective dimensionality and improve conditioning.
2. Use a fast, vectorized minimum triangle area computation.
3. Use a robust convexity check: if points are not in convex position (i.e., some points are inside the convex hull), heavily penalize the objective.
4. Multi-start Nelder-Mead: start from diverse configurations (regular polygon, random, etc.) and refine.
5. Coarse-to-fine: first run NM with loose tolerances to find basin, then re-run with tight tolerances.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized for speed.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
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
    """
    Score = min triangle area / convex hull area.
    Returns 0 if points are degenerate or not convex.
    """
    try:
        # Check for distinct points
        if len(np.unique(points, axis=0)) < N:
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
        
        # Check if all points are on the hull (strictly convex)
        # For N points, hull.vertices should have N unique indices
        if len(hull.vertices) < N:
            # Not all points are on the hull, so min triangle area might be small
            # but we still compute it. However, for maximizing min area, 
            # non-convex configurations are suboptimal.
            pass
            
        min_area = _min_triangle_area(points)
        return min_area / hull_area
    except Exception:
        return 0.0

def _objective(params):
    """
    Objective function for optimization.
    params: shape (28,) -> 14 (x,y) pairs.
    We normalize the points to have centroid (0,0) and unit norm (sum of squares = 1)
    to remove scale and translation invariance, making the landscape better conditioned.
    """
    points = params.reshape(N, 2)
    
    # Normalize: center at origin
    points = points - points.mean(axis=0)
    
    # Normalize: unit norm (sum of squared distances from origin = 1)
    norm = np.linalg.norm(points)
    if norm < 1e-10:
        return 1e10
    points = points / norm
    
    # Score is scale-invariant, so normalization doesn't change the ratio
    s = _score(points)
    
    # Penalty for non-convexity: if not all points are on the hull, penalize
    try:
        hull = ConvexHull(points)
        if len(hull.vertices) < N:
            # Penalize: the more interior points, the worse
            penalty = (N - len(hull.vertices)) * 0.1
            s -= penalty
    except Exception:
        pass
    
    if s < 0:
        return 1e10
    return -s

def _generate_initial_configs(n_configs=30):
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    regular = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(regular.flatten().copy())
    
    # 2. Regular with slight angular perturbation
    for k in range(1, 6):
        base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        perturbation = 0.05 * k * np.sin(2 * np.pi * np.arange(N) / N)
        angles = base_angles + perturbation
        angles = np.mod(angles, 2*np.pi)
        angles.sort()
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        configs.append(pts.flatten().copy())
    
    # 3. Random points in unit disk, then project to convex position? 
    # Instead, generate random points and rely on optimization to fix convexity.
    for i in range(10):
        rng = np.random.RandomState(1000 + i)
        # Random angles, random radii
        r_angles = rng.uniform(0, 2*np.pi, N)
        r_angles.sort()
        r_radii = rng.uniform(0.8, 1.2, N)
        pts = np.column_stack([r_radii * np.cos(r_angles), r_radii * np.sin(r_angles)])
        configs.append(pts.flatten().copy())
    
    # 4. Elliptic regular polygons
    for stretch in [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([stretch * np.cos(angles), np.sin(angles)])
        configs.append(pts.flatten().copy())
    
    # 5. Hexagonal-inspired: 12 points in 2 rings + 2 center? No, must be convex.
    # Instead, use a superellipse-like shape
    for n_power in [3, 4, 6]:
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = np.sign(np.cos(t)) * np.abs(np.cos(t))**(2/n_power)
        y = np.sign(np.sin(t)) * np.abs(np.sin(t))**(2/n_power)
        pts = np.column_stack([x, y])
        configs.append(pts.flatten().copy())
    
    # 6. Random with some structure: start with regular and add Gaussian noise
    for i in range(5):
        rng = np.random.RandomState(2000 + i)
        base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        noise = rng.normal(0, 0.1, N)
        angles = base_angles + noise
        angles.sort()
        radii = rng.uniform(0.9, 1.1, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        configs.append(pts.flatten().copy())
    
    return configs

def _refine_local(params, time_limit, tight=False):
    """
    Refine a configuration using Nelder-Mead.
    """
    start_time = time.time()
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e10
        return _objective(p)
    
    options = {
        'maxiter': 500 if not tight else 2000,
        'xatol': 1e-6 if not tight else 1e-10,
        'fatol': 1e-8 if not tight else 1e-12,
        'adaptive': True
    }
    
    try:
        res = minimize(
            obj,
            params,
            method='Nelder-Mead',
            options=options
        )
        return res.x
    except Exception:
        return params

def construct():
    """
    Generate 14 points to maximize min triangle area / hull area.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = -1e10
    best_params = None
    best_points = None
    
    # Phase 1: Multi-start Nelder-Mead (coarse)
    initial_configs = _generate_initial_configs(30)
    
    # Sort configs by initial score (best first)
    config_scores = []
    for cfg in initial_configs:
        s = -_objective(cfg)
        config_scores.append((s, cfg))
    config_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Refine top 15 configs
    n_to_refine = min(15, len(config_scores))
    for i in range(n_to_refine):
        if time.time() > deadline - 20:
            break
        
        s_init, cfg = config_scores[i]
        time_left = deadline - time.time() - 5
        if time_left < 2:
            break
        
        refined = _refine_local(cfg, time_left, tight=False)
        s_ref = -_objective(refined)
        
        if s_ref > best_score:
            best_score = s_ref
            best_params = refined.copy()
            # Normalize for storage
            pts = refined.reshape(N, 2)
            pts = pts - pts.mean(axis=0)
            norm = np.linalg.norm(pts)
            if norm > 1e-10:
                pts = pts / norm
            best_points = pts
    
    # Phase 2: Tight refinement of the best few
    if best_params is not None and time.time() < deadline - 10:
        # Re-run NM on the best params with tight tolerances
        time_left = deadline - time.time() - 2
        if time_left > 1:
            refined_tight = _refine_local(best_params, time_left, tight=True)
            s_tight = -_objective(refined_tight)
            if s_tight > best_score:
                best_score = s_tight
                best_params = refined_tight.copy()
                pts = refined_tight.reshape(N, 2)
                pts = pts - pts.mean(axis=0)
                norm = np.linalg.norm(pts)
                if norm > 1e-10:
                    pts = pts / norm
                best_points = pts
    
    # Phase 3: If time permits, do a few more random restarts with NM
    if time.time() < deadline - 15:
        for i in range(5):
            if time.time() > deadline - 5:
                break
            rng = np.random.RandomState(3000 + i)
            r_angles = rng.uniform(0, 2*np.pi, N)
            r_angles.sort()
            r_radii = rng.uniform(0.7, 1.3, N)
            pts = np.column_stack([r_radii * np.cos(r_angles), r_radii * np.sin(r_angles)])
            cfg = pts.flatten()
            
            time_left = deadline - time.time() - 2
            if time_left < 1:
                break
            
            refined = _refine_local(cfg, time_left, tight=False)
            s_ref = -_objective(refined)
            if s_ref > best_score:
                best_score = s_ref
                best_params = refined.copy()
                pts = refined.reshape(N, 2)
                pts = pts - pts.mean(axis=0)
                norm = np.linalg.norm(pts)
                if norm > 1e-10:
                    pts = pts / norm
                best_points = pts
    
    # Final: Normalize to unit area and return
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
