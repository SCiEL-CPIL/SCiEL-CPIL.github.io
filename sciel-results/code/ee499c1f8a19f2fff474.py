"""Heilbronn triangle problem: maximize the smallest area of a triangle formed by 11 points."""
import numpy as np
import time
from itertools import combinations

N = 11
S3 = 3 ** 0.5

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric coordinates
    h = S3 / 2
    # Check if y >= 0
    if y < -1e-10:
        return False
    # Check if below the top edge: y <= h * (1 - |x - 0.5|) / 0.5 ... 
    # Actually, the triangle is bounded by:
    # Left edge: y <= S3 * x (from (0,0) to (0.5, h))
    # Right edge: y <= S3 * (1 - x) (from (1,0) to (0.5, h))
    # Bottom edge: y >= 0
    if y > S3 * x + 1e-10:
        return False
    if y > S3 * (1 - x) + 1e-10:
        return False
    return True

def _project_to_triangle(p):
    """Project point p onto the equilateral triangle if it's outside."""
    x, y = p
    h = S3 / 2
    # Clamp y to be non-negative
    y = max(y, 0)
    # Clamp to left edge
    if y > S3 * x:
        # Project onto line y = S3 * x
        t = y / S3 if S3 != 0 else 0
        x = t
        y = S3 * x
    # Clamp to right edge
    if y > S3 * (1 - x):
        # Project onto line y = S3 * (1 - x)
        t = 1 - y / S3 if S3 != 0 else 1
        x = t
        y = S3 * (1 - x)
    return np.array([x, y])

def _initial_points():
    """Generate initial points using a grid-like distribution inside the triangle."""
    points = []
    # Corners
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, S3 / 2])
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, S3 / 4])
    points.append([0.75, S3 / 4])
    # Interior points
    points.append([0.5, S3 / 6])
    points.append([0.333, S3 / 6])
    points.append([0.667, S3 / 6])
    points.append([0.5, S3 / 3])
    return np.array(points)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    best_points = _initial_points()
    best_score = _min_area(best_points)
    
    # Local search with random perturbations
    rng = np.random.default_rng(42)
    
    # Try different starting configurations
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            points = best_points.copy()
        else:
            # Random initialization
            points = []
            for _ in range(N):
                # Random point in triangle using rejection sampling
                while True:
                    u = rng.random()
                    v = rng.random()
                    if u + v <= 1:
                        p = [u * 1.0, v * S3 / 2]
                        if _is_inside_triangle(p):
                            points.append(p)
                            break
            points = np.array(points)
        
        current_score = _min_area(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Local search
        step_size = 0.05
        max_iter = 500
        for it in range(max_iter):
            if time.time() > deadline:
                return best_points
            
            # Pick a random point to move
            idx = rng.integers(0, N)
            orig = points[idx].copy()
            
            # Random perturbation
            dx = rng.normal(0, step_size)
            dy = rng.normal(0, step_size)
            new_p = orig + np.array([dx, dy])
            
            # Project to triangle if needed
            new_p = _project_to_triangle(new_p)
            
            points[idx] = new_p
            new_score = _min_area(points)
            
            if new_score >= current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    # Reduce step size slightly
                    step_size *= 0.95
            else:
                points[idx] = orig
                step_size *= 0.99
    
    return best_points
