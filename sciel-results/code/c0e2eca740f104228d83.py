import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = points[i]
                bx, by = points[j]
                cx, cy = points[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _negative_score(params, n_outer, n_inner):
    """Negative score for minimization."""
    points = _params_to_points(params, n_outer, n_inner)
    return -_score(points)

def _params_to_points(params, n_outer, n_inner):
    """Convert parameters to point positions."""
    points = []
    # Outer ring: regular polygon with radius r_outer and angle offset
    r_outer = params[0]
    angle_offset_outer = params[1]
    for i in range(n_outer):
        angle = angle_offset_outer + 2 * np.pi * i / n_outer
        points.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
    
    # Inner ring: points at optimized radii and angles
    if n_inner > 0:
        # Use polar coordinates for inner points
        idx = 2
        for i in range(n_inner):
            r_inner = params[idx]
            angle_inner = params[idx + 1]
            idx += 2
            points.append([r_inner * np.cos(angle_inner), r_inner * np.sin(angle_inner)])
    
    return np.array(points)

def _optimize_configuration(n_outer, n_inner, bounds, maxiter=50, tol=1e-8):
    """Optimize a specific ring configuration."""
    n_params = 2 + 2 * n_inner  # r_outer, angle_offset, then (r, angle) for each inner
    
    # Set up bounds
    param_bounds = []
    param_bounds.append((0.5, 1.5))  # r_outer
    param_bounds.append((0, 2 * np.pi))  # angle_offset_outer
    for _ in range(n_inner):
        param_bounds.append((0.0, 1.0))  # r_inner
        param_bounds.append((0, 2 * np.pi))  # angle_inner
    
    # Initial guess: regular outer polygon, inner points at radius 0.5, evenly spaced
    x0 = [1.0, 0.0]
    for i in range(n_inner):
        x0.append(0.5)
        x0.append(2 * np.pi * i / n_inner)
    
    result = differential_evolution(
        _negative_score,
        bounds=param_bounds,
        args=(n_outer, n_inner),
        x0=x0,
        maxiter=maxiter,
        tol=tol,
        seed=42,
        polish=True,
        init='sobol'
    )
    
    best_params = result.x
    best_points = _params_to_points(best_params, n_outer, n_inner)
    best_score = _score(best_points)
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Try several ring configurations
    configurations = [
        (14, 0),  # All on outer ring (regular polygon)
        (12, 2),
        (10, 4),
        (8, 6),
        (7, 7),
        (6, 8),
        (5, 9),
    ]
    
    for n_outer, n_inner in configurations:
        if time.time() > deadline:
            break
        
        try:
            points, score = _optimize_configuration(n_outer, n_inner, None, maxiter=30, tol=1e-6)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception as e:
            continue
    
    # If no valid configuration found, fall back to regular 14-gon
    if best_points is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _score(best_points)
    
    return best_points
