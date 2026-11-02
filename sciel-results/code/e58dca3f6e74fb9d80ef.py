import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 120.0
SAFETY_MARGIN = 5.0

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """
    Objective function to minimize.
    We want to maximize min_triangle_area / hull_area.
    So we minimize - (min_triangle_area / hull_area).
    """
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    
    if hull_area < 1e-10:
        return 1e10 # Large penalty for degenerate hull
    
    ratio = min_area / hull_area
    return -ratio

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    global START_TIME
    start_time = time.time()
    
    best_points = None
    best_score = -float('inf')
    
    # 1. Generate diverse initial configurations
    inits = []
    
    # Regular polygon
    inits.append(_regular_polygon(N, radius=1.0))
    
    # Concentric polygons (7-6 split)
    outer = _regular_polygon(7, radius=1.0)
    inner = _regular_polygon(6, radius=0.5)
    inits.append(np.vstack([outer, inner]))
    
    # Concentric polygons (8-5 split)
    outer = _regular_polygon(8, radius=1.0)
    inner = _regular_polygon(5, radius=0.45)
    inits.append(np.vstack([outer, inner]))
    
    # Concentric polygons (9-4 split)
    outer = _regular_polygon(9, radius=1.0)
    inner = _regular_polygon(4, radius=0.4)
    inits.append(np.vstack([outer, inner]))
    
    # Concentric polygons (10-3 split)
    outer = _regular_polygon(10, radius=1.0)
    inner = _regular_polygon(3, radius=0.3)
    inits.append(np.vstack([outer, inner]))
    
    # Concentric polygons (11-2 split)
    outer = _regular_polygon(11, radius=1.0)
    inner = _regular_polygon(2, radius=0.25)
    inits.append(np.vstack([outer, inner]))
    
    # Jittered regular polygon
    for _ in range(5):
        pts = _regular_polygon(N, radius=1.0)
        noise = np.random.normal(0, 0.1, pts.shape)
        pts += noise
        inits.append(pts)
        
    # Jittered concentric polygons
    for _ in range(5):
        outer = _regular_polygon(7, radius=1.0)
        inner = _regular_polygon(6, radius=0.5)
        config = np.vstack([outer, inner])
        noise = np.random.normal(0, 0.1, config.shape)
        config += noise
        inits.append(config)
        
    # Random points in unit disk
    for _ in range(5):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)))
    
    # 2. Optimization Loop
    # Use Nelder-Mead as it is robust for non-smooth functions
    
    for init in inits:
        # Check time
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        try:
            # Reshape points to 1D for scipy
            x0 = init.flatten()
            
            # Run optimization
            res = minimize(
                _score, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000, 
                    'xatol': 1e-6, 
                    'fatol': 1e-6
                }
            )
            
            if res.success or res.fun < _score(x0):
                pts = res.x.reshape(-1, 2)
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = pts
                    
        except Exception:
            continue
            
    # If no optimization succeeded or improved, try a few random restarts with Powell
    # Or just ensure we have a valid result.
    
    if best_points is None:
        # Fallback to regular polygon
        best_points = _regular_polygon(N, radius=1.0)
        best_score = -_score(best_points)
        
    return best_points

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 120.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    ratio = min_area / hull_area
    return -ratio

def _regular_polygon(n, radius=1.0, angle_offset=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle_offset
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    best_points = None
    best_score = -float('inf')
    
    inits = []
    inits.append(_regular_polygon(N, radius=1.0))
    
    for _ in range(20):
        pts = _regular_polygon(N, radius=1.0)
        noise = np.random.normal(0, 0.2, pts.shape)
        pts += noise
        inits.append(pts)
        
    for _ in range(20):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)))

    for init in inits:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        try:
            x0 = init.flatten()
            res = minimize(
                _score, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-8
                }
            )
            if res.fun < _score(x0):
                pts = res.x.reshape(-1, 2)
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = pts
        except Exception:
            continue

    if best_points is None:
        best_points = _regular_polygon(N, radius=1.0)
        
    return best_points
# EVOLVE-BLOCK-END
