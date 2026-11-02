"""Heilbronn triangle problem: maximize minimum triangle area for 11 points."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 115.0  # seconds, safety margin before 120s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points.
    Returns the raw area (not normalized)."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # For point (x,y): 
    #   y >= 0
    #   y <= sqrt(3)*x  (left edge from (0,0) to (0.5, sqrt(3)/2))
    #   y <= sqrt(3)*(1-x)  (right edge from (1,0) to (0.5, sqrt(3)/2))
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        # Clamp y to [0, min(sqrt(3)*x, sqrt(3)*(1-x))]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        # Clamp x to [0, 1]
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _objective(x_flat):
    """Objective function for optimization: negative of min triangle area."""
    pts = x_flat.reshape(N, 2)
    if not _check_inside(pts, eps=1e-6):
        return 1e10
    return -_min_triangle_area(pts)

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    # Map to triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Using barycentric coordinates
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _initial_guess(rng):
    """Generate an initial set of 11 points."""
    pts = []
    for _ in range(N):
        pts.append(_random_point_in_triangle(rng))
    return np.array(pts)

def construct():
    """Construct 11 points in the unit equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_area = 0.0
    
    # Try multiple random starts with local optimization
    max_starts = 30
    
    for trial in range(max_starts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Generate initial points
        pts = _initial_guess(rng)
        x0 = pts.flatten()
        
        # Run optimization
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 1000}
            )
        except Exception:
            continue
        
        opt_pts = result.x.reshape(N, 2)
        opt_pts = _project_to_triangle(opt_pts)
        
        if _check_inside(opt_pts):
            area = _min_triangle_area(opt_pts)
            if area > best_area:
                best_area = area
                best_pts = opt_pts.copy()
        
        # Also try the initial points
        init_area = _min_triangle_area(pts)
        if init_area > best_area:
            best_area = init_area
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: just return random points
        pts = _initial_guess(rng)
        return pts
    
    return best_pts
