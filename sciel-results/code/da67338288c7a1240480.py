"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

# Define the equilateral triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])

def area_of_triangle(p1, p2, p3):
    """Compute the area of a triangle given three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three of the points."""
    min_area = float('inf')
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = area_of_triangle(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def objective(params):
    """Objective function to minimize the negative of the minimum triangle area."""
    points = np.array(params).reshape(-1, 2)
    
    # Check if points are inside the triangle
    # Using barycentric coordinates or simple check
    for p in points:
        # Point is inside triangle if it is on the same side of each edge
        # as the opposite vertex
        # Edge AB: y >= 0
        if p[1] < -1e-9:
            return 1e6  # Penalty
        # Edge BC: y <= sqrt(3)*(1 - x)
        if p[1] > np.sqrt(3) * (1 - p[0]) + 1e-9:
            return 1e6
        # Edge CA: y <= sqrt(3)*x
        if p[1] > np.sqrt(3) * p[0] + 1e-9:
            return 1e6
    
    min_area = min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial guess: vertices and centroid
    # Better initial guess: use a known good configuration
    # For 11 points in an equilateral triangle, a good start is a grid-like pattern
    # Or use the vertices and some interior points
    
    # Let's start with a structured initial guess
    # Vertices + 8 interior points
    init_points = [
        A, B, C,
        [1/3, 0], [2/3, 0],
        [0.5, np.sqrt(3)/6], [0.5, np.sqrt(3)/3],
        [0.25, np.sqrt(3)/4], [0.75, np.sqrt(3)/4],
        [0.5, np.sqrt(3)/2 * 0.8],
        [0.5, np.sqrt(3)/2 * 0.6]
    ]
    
    # Ensure all points are strictly inside or on boundary
    # Adjust if needed
    init_params = []
    for p in init_points:
        # Clamp to triangle
        x, y = p
        if y < 0:
            y = 0
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x)
        if y > np.sqrt(3) * x:
            y = np.sqrt(3) * x
        init_params.extend([x, y])
    
    init_params = np.array(init_params)
    
    best_result = init_params.copy()
    best_score = -objective(best_result)
    
    # Try differential evolution for global search
    # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, np.sqrt(3) / 2.0))  # y
    
    if time.time() < deadline:
        try:
            result = differential_evolution(
                objective,
                bounds,
                init=init_params,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False,
                callback=lambda xk, convergence: (time.time() > deadline)
            )
            if result.fun < -best_score:
                best_result = result.x
                best_score = -result.fun
        except Exception:
            pass
    
    # Local refinement with Nelder-Mead
    if time.time() < deadline:
        try:
            result2 = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result2.fun < -best_score:
                best_result = result2.x
                best_score = -result2.fun
        except Exception:
            pass
    
    # Final check: ensure points are valid
    points = np.array(best_result).reshape(-1, 2)
    for i, p in enumerate(points):
        x, y = p
        # Project back to triangle if slightly outside
        if y < 0:
            points[i][1] = 0
        if y > np.sqrt(3) * (1 - x):
            points[i][1] = np.sqrt(3) * (1 - x)
        if y > np.sqrt(3) * x:
            points[i][1] = np.sqrt(3) * x
        if x < 0:
            points[i][0] = 0
        if x > 1:
            points[i][0] = 1
    
    return points

# EVOLVE-BLOCK-END
