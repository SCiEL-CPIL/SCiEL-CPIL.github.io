"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

def _min_triangle_area(points):
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

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    eps = 1e-10
    
    # Constraints:
    # y >= 0
    # y <= h (max height)
    # y <= sqrt(3) * x  (left side, slope = h/0.5 = sqrt(3))
    # y <= sqrt(3) * (1 - x)  (right side)
    
    y = max(eps, y)
    y = min(y, h - eps)
    
    # From y <= sqrt(3)*x => x >= y/sqrt(3)
    x_min_left = y / np.sqrt(3)
    # From y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
    x_max_right = 1 - y / np.sqrt(3)
    
    x = max(x_min_left + eps, x)
    x = min(x_max_right - eps, x)
    
    return np.array([x, y])

def _evaluate(points):
    """Evaluate the minimum triangle area for a set of points."""
    return _min_triangle_area(points)

def _objective(x_flat, points_list):
    """Objective function for optimization (to minimize negative min area)."""
    points = np.array(points_list).reshape(-1, 2)
    # Clamp all points
    points = np.array([_clamp_to_triangle(p) for p in points])
    min_a = _min_triangle_area(points)
    return -min_a

def _local_optimize(points, max_iter=2000):
    """Perform local optimization using Nelder-Mead starting from given points."""
    x0 = np.array(points).flatten()
    
    result = minimize(
        _objective,
        x0,
        args=(points,),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10}
    )
    
    opt_points = np.array(result.x).reshape(-1, 2)
    opt_points = np.array([_clamp_to_triangle(p) for p in opt_points])
    return opt_points, _evaluate(opt_points)

def _symmetric_initial():
    """Create a symmetric initial configuration with 11 points."""
    h = np.sqrt(3) / 2
    # Vertices + centroid + 7 symmetric points
    points = [
        [0.0, 0.0],      # vertex 1
        [1.0, 0.0],      # vertex 2
        [0.5, h],         # vertex 3
        [0.5, h/3],       # centroid
        [0.25, h/6],
        [0.75, h/6],
        [0.5, h/2],
        [0.125, h/12],
        [0.875, h/12],
        [0.375, h/4],
        [0.625, h/4],
    ]
    return np.array(points)

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s safety margin
    
    h = np.sqrt(3) / 2
    
    # Start with symmetric configuration
    best_points = _symmetric_initial()
    best_score = _evaluate(best_points)
    
    # First, do local optimization from symmetric start
    if time.time() < deadline - 30:
        opt_points, opt_score = _local_optimize(best_points, max_iter=3000)
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
    
    # Now use differential evolution for global search
    # Define bounds for each coordinate
    bounds = [(0, 1)] * (2 * N)
    
    def _de_objective(x_flat):
        points = np.array(x_flat).reshape(-1, 2)
        points = np.array([_clamp_to_triangle(p) for p in points])
        return -_min_triangle_area(points)
    
    # Run differential evolution with multiple restarts
    np.random.seed(42)
    
    # DE run 1: broad search
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _de_objective,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-8,
                popsize=15,
                mutation=(0.5, 1.5),
                recombination=0.7,
                polish=True,
                init='sobol',
                time_limit=deadline - time.time() - 60
            )
            de_points = np.array(de_result.x).reshape(-1, 2)
            de_points = np.array([_clamp_to_triangle(p) for p in de_points])
            de_score = _evaluate(de_points)
            if de_score > best_score:
                best_points = de_points
                best_score = de_score
        except Exception:
            pass
    
    # DE run 2: different seed and parameters
    if time.time() < deadline - 30:
        try:
            de_result2 = differential_evolution(
                _de_objective,
                bounds,
                seed=123,
                maxiter=150,
                tol=1e-8,
                popsize=20,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True,
                init='latinhypercube',
                time_limit=deadline - time.time() - 30
            )
            de_points2 = np.array(de_result2.x).reshape(-1, 2)
            de_points2 = np.array([_clamp_to_triangle(p) for p in de_points2])
            de_score2 = _evaluate(de_points2)
            if de_score2 > best_score:
                best_points = de_points2
                best_score = de_score2
        except Exception:
            pass
    
    # Final local polish from best found
    if time.time() < deadline - 5:
        opt_points, opt_score = _local_optimize(best_points, max_iter=5000)
        if opt_score > best_score:
            best_points = opt_points
            best_score = opt_score
    
    # Final check: ensure all points are inside the triangle
    best_points = np.array([_clamp_to_triangle(p) for p in best_points])
    
    return best_points

# EVOLVE-BLOCK-END
