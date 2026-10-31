"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

def _min_triangle_area(points):
    """Compute the minimum triangle area over all triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _ratio(points):
    """Compute the score: min triangle area / convex hull area."""
    ma = _min_triangle_area(points)
    ha = _convex_hull_area(points)
    if ha < 1e-10:
        return 0.0
    return ma / ha

def _points_from_angles(angles, rx=1.0, ry=1.0):
    """Generate points on an ellipse from angles."""
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    return np.column_stack([rx * cos_a, ry * sin_a])

def _neg_ratio_from_params(params, rx=1.0, ry=1.0):
    """Objective: negative ratio from angular parameters."""
    angles = params
    points = _points_from_angles(angles, rx, ry)
    return -_ratio(points)

def construct():
    """
    Construct 13 points to maximize min triangle area / convex hull area.
    New approach: Parameterize points on an ellipse, optimize angles + shape.
    """
    best_score = 0.0
    best_points = None
    
    start_time = time.time()
    timeout = 110.0
    
    # Strategy: Optimize angular positions on an ellipse
    # Parameters: N angles (sorted) + 1 shape parameter (ry/rx ratio)
    # We'll use differential evolution for global search, then local refinement
    
    def _full_objective(params):
        """Objective with shape parameter included."""
        angles = params[:-1]
        shape = params[-1]
        # Ensure angles are sorted and in [0, 2*pi)
        angles = np.sort(angles)
        # Center angles: shift so mean is 0 for symmetry
        angles = angles - np.mean(angles)
        points = _points_from_angles(angles, rx=1.0, ry=shape)
        return -_ratio(points)
    
    # Bounds: angles in [-pi, pi], shape in [0.5, 2.0]
    bounds = [(-np.pi, np.pi)] * N + [(0.5, 2.0)]
    
    # Initial guess: equally spaced angles on a circle
    base_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    base_angles = base_angles - np.mean(base_angles)
    x0 = np.concatenate([base_angles, [1.0]])
    
    # Run differential evolution with limited population for speed
    try:
        if time.time() - start_time < timeout - 60:
            de_res = differential_evolution(
                _full_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.2),
                recombination=0.8,
                polish=True
            )
            if time.time() - start_time < timeout - 20:
                score_de = -de_res.fun
                if score_de > best_score:
                    best_score = score_de
                    angles_opt = np.sort(de_res.x[:-1])
                    shape_opt = de_res.x[-1]
                    best_points = _points_from_angles(angles_opt, rx=1.0, ry=shape_opt)
    except Exception:
        pass
    
    # If DE didn't run or failed, use multi-start local optimization
    if best_points is None or time.time() - start_time < timeout - 30:
        # Multiple initial configurations
        np.random.seed(42)
        
        configs = []
        # 1. Regular polygon on circle
        angles1 = np.linspace(0, 2*np.pi, N, endpoint=False) - np.mean(np.linspace(0, 2*np.pi, N, endpoint=False))
        configs.append((angles1, 1.0))
        
        # 2. Slightly perturbed
        angles2 = angles1 + np.random.normal(0, 0.05, N)
        configs.append((angles2, 1.0))
        
        # 3. Elliptical with different aspect ratios
        for shape in [0.7, 0.85, 1.15, 1.3]:
            configs.append((angles1, shape))
        
        # 4. Random angular perturbations
        for _ in range(4):
            angles_r = np.sort(np.random.uniform(-np.pi, np.pi, N))
            configs.append((angles_r, np.random.uniform(0.7, 1.3)))
        
        for angles_init, shape_init in configs:
            if time.time() - start_time > timeout - 10:
                break
            
            x0 = np.concatenate([angles_init, [shape_init]])
            
            try:
                res = minimize(
                    _full_objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'maxiter': 3000,
                        'maxfev': 3000
                    }
                )
                
                if time.time() - start_time > timeout - 5:
                    break
                    
                score = -res.fun
                if score > best_score:
                    best_score = score
                    angles_opt = np.sort(res.x[:-1])
                    shape_opt = res.x[-1]
                    best_points = _points_from_angles(angles_opt, rx=1.0, ry=shape_opt)
            except Exception:
                continue
    
    # Fallback: regular polygon
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = _points_from_angles(angles)
    
    # Normalize to unit area convex hull
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        scale = 1.0 / np.sqrt(hull_area)
        best_points = best_points * scale
    
    return best_points
# EVOLVE-BLOCK-END
