import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
DEADLINE = 120.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points.
    points: (N, 2) array
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized calculation for all triples is memory heavy for large N, 
    # but N=13 is small (286 triangles).
    # We can loop or use broadcasting.
    # Broadcasting: 
    # i, j, k indices
    # Let's just use a loop for simplicity and clarity, it's fast enough for N=13.
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
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
    
    # Check if points are collinear or identical
    # ConvexHull raises QhullError if degenerate
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
    
    # Initializations
    inits = []
    
    # 1. Regular polygon
    inits.append(_regular_polygon(N, radius=1.0))
    
    # 2. Regular polygon with different radii/offsets (scale invariance makes radius less critical, but jitter helps)
    for _ in range(10):
        # Jittered regular polygon
        pts = _regular_polygon(N, radius=1.0)
        noise = np.random.normal(0, 0.1, pts.shape)
        pts += noise
        inits.append(pts)
        
    # 3. Random points in unit disk
    for _ in range(10):
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        inits.append(np.column_stack((x, y)))
        
    # 4. Grid-like points
    # 13 points in a 4x4 grid minus 3 corners?
    # Just random is probably fine for multi-start.
    
    # Optimization Loop
    # Use Nelder-Mead as it is robust for non-smooth functions
    # Powell is also good. Let's try Nelder-Mead first.
    
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
