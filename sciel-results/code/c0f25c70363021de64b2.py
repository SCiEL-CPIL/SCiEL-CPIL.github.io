import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import sys

N = 13

def _triangle_areas(points):
    """Compute all triangle areas for a set of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    return np.array(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _negative_score(points_flat):
    """Objective function for optimization (minimize negative score)."""
    points = points_flat.reshape(N, 2)
    score = _score(points)
    return -score

def _regular_polygon(n, radius=1.0):
    """Generate a regular n-gon centered at origin."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _circle_with_interior(n, radius=1.0):
    """Generate points on a circle with some interior points."""
    # Start with regular polygon
    pts = _regular_polygon(n, radius)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Initial configurations
    initial_configs = []
    
    # Regular 13-gon
    initial_configs.append(_regular_polygon(N, 1.0))
    
    # Regular 13-gon with different radius
    initial_configs.append(_regular_polygon(N, 0.8))
    initial_configs.append(_regular_polygon(N, 1.2))
    
    # Grid-like distribution
    # 4x4 grid minus one corner, scaled
    grid = []
    for i in range(4):
        for j in range(4):
            if len(grid) < N:
                grid.append([i*0.5, j*0.5])
    initial_configs.append(np.array(grid[:N]))
    
    # Random configurations
    rng = np.random.default_rng(42)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        pts = rng.random((N, 2))
        # Scale to unit area roughly
        pts = pts * 1.0
        initial_configs.append(pts)
    
    # Evaluate initial configs
    for config in initial_configs:
        score = _score(config)
        if score > best_score:
            best_score = score
            best_points = config.copy()
    
    # Local optimization from each initial config
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _negative_score,
                config.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < -best_score:
                opt_points = result.x.reshape(N, 2)
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    # If we have time, do more random restarts
    while time.time() < deadline - 5:
        rng = np.random.default_rng()
        pts = rng.random((N, 2))
        
        try:
            result = minimize(
                _negative_score,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result.success:
                opt_points = result.x.reshape(N, 2)
                opt_score = _score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to regular polygon
        best_points = _regular_polygon(N, 1.0)
    
    return best_points
