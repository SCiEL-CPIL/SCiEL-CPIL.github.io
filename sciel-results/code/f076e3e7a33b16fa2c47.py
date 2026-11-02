"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
from itertools import combinations


def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three 2D points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by 3 of the given points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area


def _is_inside_equilateral(p):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    x, y = p
    h = np.sqrt(3) / 2
    # y >= 0
    if y < -1e-10:
        return False
    # y <= h
    if y > h + 1e-10:
        return False
    # x >= 0
    if x < -1e-10:
        return False
    # x <= 1
    if x > 1 + 1e-10:
        return False
    # Left edge: from (0,0) to (0.5, h). Line: y = sqrt(3)*x => x >= y/sqrt(3)
    if x < y / np.sqrt(3) - 1e-10:
        return False
    # Right edge: from (1,0) to (0.5, h). Line: y = -sqrt(3)*(x-1) => x <= 1 - y/sqrt(3)
    if x > 1 - y / np.sqrt(3) + 1e-10:
        return False
    return True


def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle."""
    x, y = p
    h = np.sqrt(3) / 2
    # Clamp y
    y = np.clip(y, 0, h)
    # Clamp x based on y
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    x = np.clip(x, x_min, x_max)
    return np.array([x, y])


def _objective(flat_params):
    """Objective function for optimization: negative of min triangle area."""
    points = flat_params.reshape((N, 2))
    # Clamp all points to be inside the triangle
    clamped = np.array([_clamp_to_triangle(p) for p in points])
    min_area = _min_triangle_area(clamped)
    return -min_area


def construct():
    # Initial guess: use a grid-like pattern inside the triangle
    # We'll use differential evolution to optimize the positions
    
    # Bounds for each coordinate
    h = np.sqrt(3) / 2
    bounds = []
    for i in range(N):
        # x can range from 0 to 1, y from 0 to h
        # But we'll use tighter bounds based on the triangle shape
        # For simplicity, use [0,1] for x and [0, h] for y, then clamp
        bounds.append((0, 1))
        bounds.append((0, h))
    
    # Run differential evolution
    result = differential_evolution(
        _objective,
        bounds,
        seed=42,
        maxiter=1000,
        popsize=20,
        tol=1e-8,
        polish=True,
        workers=-1
    )
    
    points = result.x.reshape((N, 2))
    # Clamp to ensure all points are inside
    points = np.array([_clamp_to_triangle(p) for p in points])
    
    return points
# EVOLVE-BLOCK-END
