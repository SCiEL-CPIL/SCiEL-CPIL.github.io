"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area(points):
    """Return the minimum triangle area among all triplets."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _is_inside_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    x, y = p
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Edge 1: from (0,0) to (1,0) -> y >= 0
    if y < -tol:
        return False
    # Edge 2: from (1,0) to (0.5, sqrt(3)/2) -> line: y = -sqrt(3)(x - 1)
    if y > np.sqrt(3) * (1 - x) + tol:
        return False
    # Edge 3: from (0.5, sqrt(3)/2) to (0,0) -> line: y = sqrt(3)*x
    if y > np.sqrt(3) * x + tol:
        return False
    return True

def _project_to_triangle(p):
    """Project a point to the nearest point inside the triangle."""
    x, y = p
    cx, cy = 0.5, np.sqrt(3) / 6
    if not _is_inside_triangle(p):
        # Move toward centroid iteratively
        for _ in range(50):
            if _is_inside_triangle([x, y]):
                break
            x = 0.5 * (x + cx)
            y = 0.5 * (y + cy)
    return [x, y]

def _project_points_array(points):
    """Project an array of points to the triangle."""
    return np.array([_project_to_triangle(p) for p in points])

def _objective(x):
    """Objective function: negative of minimum area (we minimize this)."""
    points = x.reshape(N, 2)
    proj_points = _project_points_array(points)
    ma = _min_area(proj_points)
    return -ma

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the minimum triangle area."""
    
    # Use differential evolution for global optimization
    # Bounds for each coordinate
    # x: [0, 1], y: [0, sqrt(3)/2]
    bounds = [(0, 1)] * N + [(0, np.sqrt(3)/2)] * N
    
    # Better approach: use separate bounds for x and y
    # x_i in [0, 1], y_i in [0, sqrt(3)/2]
    # But we need to ensure points are inside the triangle
    
    # Let's use a different parameterization
    # Use barycentric coordinates: each point has 3 barycentric coords that sum to 1
    # Or use a 2D parameterization with projection
    
    # Actually, let's just use the 2D coordinates and rely on projection
    # But for differential evolution, we need bounds
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, np.sqrt(3)/2))  # y
    
    # Run differential evolution
    result = differential_evolution(
        _objective,
        bounds,
        maxiter=200,
        tol=1e-6,
        seed=42,
        polish=False,
        init='sobol'
    )
    
    # Get the best points
    best_points = result.x.reshape(N, 2)
    best_points = _project_points_array(best_points)
    
    # Try a few more random restarts with differential evolution
    best_score = _min_area(best_points)
    best_points_final = best_points
    
    for seed in range(5):
        np.random.seed(seed * 100 + 7)
        try:
            result = differential_evolution(
                _objective,
                bounds,
                maxiter=150,
                tol=1e-6,
                seed=seed * 100 + 7,
                polish=False,
                init='latinhypercube'
            )
            points = result.x.reshape(N, 2)
            points = _project_points_array(points)
            score = _min_area(points)
            if score > best_score:
                best_score = score
                best_points_final = points
        except Exception:
            continue
    
    # Ensure all points are strictly inside
    final_points = []
    cx, cy = 0.5, np.sqrt(3) / 6
    for i in range(N):
        p = best_points_final[i]
        # Slight inward shift to ensure strictly inside
        p_shifted = 0.999 * p + 0.001 * np.array([cx, cy])
        if not _is_inside_triangle(p_shifted):
            p_shifted = p
        if not _is_inside_triangle(p_shifted):
            p_shifted = np.array([cx, cy])
        final_points.append(p_shifted)
    
    return np.array(final_points)
# EVOLVE-BLOCK-END
