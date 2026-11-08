"""Heilbronn convex regions n=14: maximize minimum triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14

# EVOLVE-BLOCK-START
import random
import math

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 1e-10

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_a

def _objective(x):
    """Objective function for optimization: negative score (we maximize score)."""
    points = x.reshape(N, 2)
    # Ensure points are distinct and not degenerate
    s = _score(points)
    return -s

def _random_initial(rng, n=N):
    """Generate a random initial configuration."""
    return np.array([[rng.random(), rng.random()] for _ in range(n)])

def _grid_initial(n=N):
    """Generate a grid-like initial configuration."""
    # Try to space points evenly
    side = int(math.ceil(math.sqrt(n)))
    points = []
    for i in range(n):
        row = i // side
        col = i % side
        points.append([(col + 0.5) / side, (row + 0.5) / side])
    return np.array(points[:n])

def _optimize_from_initial(x0, method='L-BFGS-B', maxiter=1000, timeout=10):
    """Optimize from an initial point with a time limit."""
    start_time = time.time()
    
    # Use L-BFGS-B with bounds
    bounds = [(0, 1)] * (2 * N)
    
    # Simple local optimization
    res = minimize(_objective, x0, method=method, bounds=bounds, 
                   options={'maxiter': maxiter, 'maxfun': 5000})
    
    elapsed = time.time() - start_time
    return res.x, res.fun, elapsed

def construct():
    """Construct 14 points to maximize minimum triangle area / hull area."""
    global_start = time.time()
    deadline = global_start + 140  # Safety margin before 150s limit
    
    best_points = None
    best_score = -float('inf')
    
    # Multiple strategies
    strategies = []
    
    # Strategy 1: Grid-based initial
    grid_pts = _grid_initial(N)
    strategies.append(('grid', grid_pts))
    
    # Strategy 2: Random initializations
    rng = random.Random(42)
    for i in range(10):
        rand_pts = _random_initial(rng, N)
        strategies.append((f'rand_{i}', rand_pts))
    
    # Strategy 3: Circle-based initial
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    circle_pts = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    strategies.append(('circle', circle_pts))
    
    # Evaluate all initial strategies
    for name, pts in strategies:
        if time.time() > deadline - 5:
            break
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    # Local optimization from best initial
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            opt_x, opt_fun, elapsed = _optimize_from_initial(x0, method='L-BFGS-B', maxiter=500, timeout=10)
            opt_pts = opt_x.reshape(N, 2)
            opt_s = _score(opt_pts)
            if opt_s > best_score:
                best_score = opt_s
                best_points = opt_pts.copy()
        except Exception:
            pass
    
    # Try a few more random restarts with optimization if time permits
    rng2 = random.Random(123)
    while time.time() < deadline - 15:
        rand_pts = _random_initial(rng2, N)
        s = _score(rand_pts)
        if s > best_score * 0.9:  # Only optimize promising starts
            x0 = rand_pts.flatten()
            try:
                opt_x, opt_fun, elapsed = _optimize_from_initial(x0, method='L-BFGS-B', maxiter=300, timeout=5)
                opt_pts = opt_x.reshape(N, 2)
                opt_s = _score(opt_pts)
                if opt_s > best_score:
                    best_score = opt_s
                    best_points = opt_pts.copy()
            except Exception:
                pass
        else:
            # Just evaluate without optimization
            if s > best_score:
                best_score = s
                best_points = rand_pts.copy()
    
    # Final check: ensure we have a valid result
    if best_points is None:
        # Fallback to grid
        best_points = _grid_initial(N)
    
    return best_points
# EVOLVE-BLOCK-END
