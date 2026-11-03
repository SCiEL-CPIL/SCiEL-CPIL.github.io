import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
SQRT3 = np.sqrt(3)
TRIS_AREA = SQRT3 / 4.0 # Area of the unit equilateral triangle (side 1)
DEADLINE = 110.0 # Safety margin

def _triangle_area(p1, p2, p3):
    # Shoelace formula for area of triangle given 3 points
    # 0.5 * | (x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)) |
    return 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))

def _is_inside_triangle(p):
    # Check if point p is inside the triangle (0,0), (1,0), (0.5, sqrt(3)/2)
    # y >= 0
    # y <= sqrt(3) * x
    # y <= -sqrt(3) * (x - 1)
    x, y = p
    if y < -1e-9: return False
    if y > SQRT3 * x + 1e-9: return False
    if y > -SQRT3 * (x - 1) + 1e-9: return False
    return True

def _min_triangle_area(points):
    # points is an array of shape (N, 2)
    min_area = float('inf')
    # Iterate over all combinations of 3 points
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _objective_flat(x):
    # x is a flat array of length 2*N
    points = x.reshape(N, 2)
    
    # Check validity
    for p in points:
        if not _is_inside_triangle(p):
            # Return a large negative value (since we maximize, we minimize -area)
            # Or return a penalty that makes the optimizer avoid this
            return -100.0 
            
    area = _min_triangle_area(points)
    return -area # We want to maximize area, so minimize negative area

def construct():
    start_time = time.time()
    
    # Initial guesses
    # 1. Grid-like pattern
    # 2. Random
    # 3. Symmetric center
    
    # Bounds for DE: x in [0, 1], y in [0, SQRT3/2]
    bounds = [(0, 1), (0, SQRT3/2)] * N
    
    best_points = None
    best_score = 0.0
    
    # Strategy: Run Differential Evolution with limited iterations
    # Then refine with Nelder-Mead
    
    # We will try a few different initial strategies or just one robust DE run
    # DE is stochastic, so we can run it once with a decent population
    
    try:
        # Use differential_evolution to find a good starting point
        # maxiter=100, popsize=15 is a reasonable balance
        # We need to be careful with the time limit
        
        res = differential_evolution(
            _objective_flat, 
            bounds, 
            maxiter=50, 
            popsize=10, 
            seed=42,
            tol=1e-6,
            init='sobol' # Good sampling
        )
        
        current_time = time.time()
        if current_time > DEADLINE:
             # Return whatever we have, or a fallback
             # If DE failed or timed out, we might not have res
             pass
        else:
            best_points = res.x.reshape(N, 2)
            best_score = -res.fun
            
            # Refine the result
            # Use Nelder-Mead for local optimization
            res2 = minimize(
                _objective_flat, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res2.success or res2.fun < res.fun:
                best_points = res2.x.reshape(N, 2)
                best_score = -res2.fun
                
            current_time = time.time()
            
            # If we have time, try another restart with a different seed
            if current_time < DEADLINE - 20:
                 res3 = differential_evolution(
                    _objective_flat, 
                    bounds, 
                    maxiter=50, 
                    popsize=10, 
                    seed=123,
                    tol=1e-6,
                    init='sobol'
                )
                 if -res3.fun > best_score:
                     best_points = res3.x.reshape(N, 2)
                     best_score = -res3.fun
                     
                     # Refine again
                     res4 = minimize(
                        _objective_flat, 
                        best_points.flatten(), 
                        method='Nelder-Mead', 
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                    )
                     if res4.success or res4.fun < res3.fun:
                         best_points = res4.x.reshape(N, 2)
                         best_score = -res4.fun
                         
    except Exception as e:
        # Fallback to a simple deterministic pattern if optimization fails
        # E.g. regular grid
        best_points = np.zeros((N, 2))
        # Just place them in a rough grid to ensure validity
        # This is a very weak fallback, but better than crashing
        import random
        rng = random.Random(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, SQRT3 / 2 * v])
        best_points = np.array(points)
        best_score = _min_triangle_area(best_points)

    return best_points

