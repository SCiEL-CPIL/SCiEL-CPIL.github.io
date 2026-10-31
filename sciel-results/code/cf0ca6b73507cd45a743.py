"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations

def _triangle_area(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p):
    """Check if point p is inside the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    # The triangle has vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # Barycentric coordinates or half-plane tests
    # Edge 1: from (0,0) to (1,0): y >= 0
    if y < -1e-10:
        return False
    # Edge 2: from (1,0) to (0.5, sqrt(3)/2): line is y = -sqrt(3)(x - 1)
    # So y <= -sqrt(3)(x - 1) => y + sqrt(3)x - sqrt(3) <= 0
    if y + np.sqrt(3) * x - np.sqrt(3) > 1e-10:
        return False
    # Edge 3: from (0.5, sqrt(3)/2) to (0,0): line is y = sqrt(3)x
    # So y <= sqrt(3)x
    if y - np.sqrt(3) * x > 1e-10:
        return False
    return True

def _objective(x_flat):
    """Objective function for optimization: negative of the minimum triangle area.
    We want to maximize the minimum area, so we minimize the negative."""
    points = x_flat.reshape(-1, 2)
    
    # Check if all points are inside the triangle
    for p in points:
        if not _inside_triangle(p):
            return 1e10  # Large penalty
    
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    """Construct 11 points inside the equilateral triangle to maximize the smallest triangle area."""
    
    # Define the bounds for each coordinate
    # x: [0, 1], y: [0, sqrt(3)/2]
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # Use differential evolution to find the optimal configuration
    # This is a global optimization method that works well for non-convex problems
    
    # Initial best result from the seed program (for comparison)
    def _seed_program():
        import random
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        return np.array(points)
    
    seed_points = _seed_program()
    seed_min_area = _min_triangle_area(seed_points)
    
    # Run differential evolution
    result = differential_evolution(
        _objective,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-8,
        polish=True,
        init='sobol'
    )
    
    best_points = result.x.reshape(-1, 2)
    best_min_area = -result.fun
    
    # Compare with seed
    if best_min_area < seed_min_area:
        best_points = seed_points
    
    return best_points
# EVOLVE-BLOCK-END
