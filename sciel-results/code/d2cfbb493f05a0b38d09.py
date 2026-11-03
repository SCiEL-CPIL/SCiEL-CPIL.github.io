"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle


def point_in_triangle(p, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the given points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                area = 0.5 * abs(
                    (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                    (points[j][1] - points[i][1]) * (points[k][0] - points[i][0])
                )
                if area < min_area:
                    min_area = area
    return min_area


def negative_min_area_flat(x):
    """
    Objective function for differential_evolution.
    x is a flat array of length 2*N.
    Returns the negative of the minimum triangle area (to be minimized).
    If any point is outside the triangle, return a large penalty.
    """
    points = x.reshape(N, 2)
    for p in points:
        if not point_in_triangle(p, eps=1e-7):
            return 1e10  # penalty
    return -min_triangle_area(points)


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Initial guess: deterministic hierarchical subdivision
    # Divide the triangle into 11 sub-regions and place points at centroids
    # A simple approach: use a grid-like arrangement
    # For an equilateral triangle, we can use a triangular lattice pattern
    # Let's use 11 points in a pattern:
    # Row 0: 1 point
    # Row 1: 2 points
    # Row 2: 3 points
    # Row 3: 3 points
    # Row 4: 2 points
    # Total: 11 points
    
    # Actually, let's use a simpler initial guess: evenly spaced points
    # Using barycentric coordinates for a triangular grid
    # We'll use a 4x4 triangular grid and select 11 points
    
    # Let's use a different approach: start with a regular hexagonal-like pattern
    # centered in the triangle, plus some boundary points
    
    # Simple initial guess: use numpy linspace in barycentric coordinates
    # Convert barycentric (u, v, w) with u+v+w=1 to Cartesian
    # x = u*V0 + v*V1 + w*V2
    
    # Create a grid of barycentric coordinates
    # For a triangular arrangement, we can use:
    # Level 0: (1,0,0), (0,1,0), (0,0,1) - vertices
    # Level 1: (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2) - midpoints
    # Level 2: various combinations
    
    # Let's just use a random initial guess with fixed seed for reproducibility
    # and let differential_evolution handle the optimization
    
    # Bounds for each coordinate
    # x ranges from 0 to 1
    # y ranges from 0 to sqrt(3)/2
    bounds = [(0, 1), (0, np.sqrt(3) / 2)] * N
    
    # Run differential evolution
    result = differential_evolution(
        negative_min_area_flat,
        bounds,
        seed=42,
        maxiter=200,
        popsize=15,
        tol=1e-8,
        polish=True,
        callback=None
    )
    
    # Check if we're still within time limit
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return the best found so far
        best_points = result.x.reshape(N, 2)
        # Verify all points are inside the triangle
        valid = all(point_in_triangle(p) for p in best_points)
        if valid:
            return best_points
        else:
            # Fallback to a simple valid configuration
            return _fallback_configuration()
    
    best_points = result.x.reshape(N, 2)
    
    # Verify all points are inside the triangle
    valid = all(point_in_triangle(p) for p in best_points)
    if not valid:
        return _fallback_configuration()
    
    return best_points


def _fallback_configuration():
    """Return a simple valid configuration of 11 points inside the triangle."""
    # Use a triangular lattice pattern
    points = []
    
    # Vertices
    points.append([0.1, 0.05])
    points.append([0.9, 0.05])
    points.append([0.5, np.sqrt(3) / 2 * 0.85])
    
    # Midpoints of edges
    points.append([0.5, 0.05])
    points.append([0.3, np.sqrt(3) / 4])
    points.append([0.7, np.sqrt(3) / 4])
    
    # Interior points
    points.append([0.5, np.sqrt(3) / 6])
    points.append([0.35, np.sqrt(3) / 3])
    points.append([0.65, np.sqrt(3) / 3])
    points.append([0.45, np.sqrt(3) / 2 * 0.6])
    points.append([0.55, np.sqrt(3) / 2 * 0.6])
    
    return np.array(points)
# EVOLVE-BLOCK-END
