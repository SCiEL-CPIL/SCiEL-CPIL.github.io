import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution

N = 13

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    points: numpy array of shape (N, 2)
    """
    n = len(points)
    min_area = float('inf')
    # For N=13, 286 triangles is small, loop is fine
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            for k in range(j + 1, n):
                p3 = points[k]
                # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """
    Compute the area of the convex hull of the given points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri_area = _min_triangle_area(points)
    return min_tri_area / hull_area

def _objective(x):
    """
    Objective function for minimization (negative score).
    """
    points = x.reshape(N, 2)
    return -_score(points)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Bounds for differential evolution
    # We can restrict to [-1, 1] as the problem is scale invariant
    bounds = [(-1, 1)] * (2 * N)
    
    best_x = init_points.flatten()
    best_score = _score(init_points)
    
    # Use Differential Evolution
    # popsize=10, maxiter=100 is a good balance for 26 vars
    # We will monitor time.
    # DE in scipy doesn't have a direct time limit, so we run a batch and check.
    
    try:
        # We can use a callback to check time, but DE's callback gets iteration count.
        # Simpler: Run DE with a reasonable maxiter.
        # 26 vars, popsize 10 -> 260 evals/iter.
        # If eval takes 1ms, 1 iter is 0.26s.
        # 100 iters is 26s.
        # 300 iters is 78s.
        # Let's try 200 iters.
        
        result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=15,
            tol=1e-6,
            polish=True # Local refinement at the end
        )
        
        if result.success or result.fun < _objective(best_x):
            best_x = result.x
            best_score = -result.fun
            
    except Exception as e:
        # Fallback to initial guess if something goes wrong
        pass
    
    # Check if we still have time for a local refinement if DE didn't polish well
    # Or if DE was cut short (it won't be, but just in case)
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Try a local refinement using SLSQP on the best point found
        from scipy.optimize import minimize
        
        # Ensure the points are not degenerate
        x0 = best_x.copy()
        
        # SLSQP can be slow, use a tight maxiter
        res = minimize(
            _objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-9}
        )
        
        if res.success and -res.fun > best_score:
            best_score = -res.fun
            best_x = res.x
            
    # Return the best points found
    return best_x.reshape(N, 2)
