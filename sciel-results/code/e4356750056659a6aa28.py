"""Rewritten program for Heilbronn convex regions n=14; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def _triangle_areas(points):
    """Compute all triangle areas for a set of points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area = 0.5 * |cross product|
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    # Sort by x, then y
    points = np.array(points)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    m = len(hull)
    for i in range(m):
        j = (i + 1) % m
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area


def _negative_score(points):
    """Negative score for minimization."""
    return -_score(points)


def construct():
    # Start with a regular polygon-like configuration
    # Use a grid or random initialization, then optimize
    
    # Initialize with points in a circular arrangement
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = 0.5 + 0.4 * np.cos(theta)
    y = 0.5 + 0.4 * np.sin(theta)
    x0 = np.column_stack((x, y)).flatten()
    
    # Try multiple random restarts
    best_x = x0.copy()
    best_score = _score(best_x.reshape(N, 2))
    
    np.random.seed(42)
    
    # Try a few random initializations
    for restart in range(3):
        if restart == 0:
            x_init = x0.copy()
        else:
            # Random points in unit square
            x_init = np.random.rand(N * 2)
        
        # Optimize using Nelder-Mead
        result = minimize(
            _negative_score,
            x_init,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        
        score = _score(result.x.reshape(N, 2))
        if score > best_score:
            best_score = score
            best_x = result.x.copy()
    
    # Additional local refinement
    result = minimize(
        _negative_score,
        best_x,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    points = result.x.reshape(N, 2)
    score = _score(points)
    
    # If the score is not better than the initial, try a different approach
    # Use a more structured initialization: points on concentric circles or grid
    
    # Try grid-based initialization
    # Place points in a 4x4 grid with some jitter
    grid_x = []
    grid_y = []
    for i in range(4):
        for j in range(4):
            if len(grid_x) < N:
                grid_x.append(0.1 + 0.8 * i / 3.0)
                grid_y.append(0.1 + 0.8 * j / 3.0)
    
    x_grid = np.column_stack((grid_x, grid_y)).flatten()
    result_grid = minimize(
        _negative_score,
        x_grid,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    score_grid = _score(result_grid.x.reshape(N, 2))
    
    if score_grid > score:
        points = result_grid.x.reshape(N, 2)
        score = score_grid
    
    # Final refinement
    result_final = minimize(
        _negative_score,
        points.flatten(),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11}
    )
    
    points = result_final.x.reshape(N, 2)
    
    return points
# EVOLVE-BLOCK-END
