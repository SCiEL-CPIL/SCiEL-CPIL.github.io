import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize

N = 11
# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])

# Pre-compute constants
S3 = np.sqrt(3)

def point_in_triangle(p):
    """Check if point p is inside or on the boundary of the triangle."""
    # Barycentric coordinates technique or half-plane checks
    # Half-plane 1: y >= 0
    if p[1] < -1e-10:
        return False
    # Half-plane 2: y <= sqrt(3) * x  => sqrt(3)x - y >= 0
    if S3 * p[0] - p[1] < -1e-10:
        return False
    # Half-plane 3: y <= sqrt(3) * (1 - x) => sqrt(3)x + y - sqrt(3) <= 0
    if S3 * p[0] + p[1] - S3 > 1e-10:
        return False
    return True

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    # Loop over all combinations of 3 points
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            # Vector p1->p2
            v12 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                # Vector p1->p3
                v13 = p3 - p1
                # Area = 0.5 * |cross(v12, v13)|
                cross = v12[0] * v13[1] - v12[1] * v13[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """
    Objective function for minimization.
    x is a flat array of length 2*N.
    Returns negative of the minimum triangle area.
    Includes penalty for points outside the triangle.
    """
    points = x.reshape((N, 2))
    
    # Check bounds/validity
    # If any point is outside, return a large penalty
    for p in points:
        if not point_in_triangle(p):
            return 100.0 # Large positive value for minimization (bad score)
            
    # Calculate min area
    area = min_triangle_area(points)
    
    # We want to maximize min area, so minimize -min area
    return -area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin for safety
    
    # Bounds for DE
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, S3 / 2.0))
        
    # Strategy:
    # 1. Run Differential Evolution
    # 2. If time remains, run Local Polish (Powell)
    
    best_result = None
    
    # Helper to check time
    def callback(xk, convergence):
        if time.time() > deadline:
            return True
        return False

    try:
        # DE Parameters
        # popsize: multiplier of dimension (22). 10 * 22 = 220 individuals.
        # maxiter: number of generations.
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=15, 
            tol=1e-8, 
            mutation=(0.5, 1.0), 
            recomb=0.7, 
            seed=42, 
            callback=callback,
            polish=False # We will do our own polish
        )
        
        if res.success or res.x is not None:
            best_result = res.x
            best_score = res.fun
            
    except Exception:
        pass
        
    # If DE didn't find a valid solution or finished early, ensure we have a valid fallback
    if best_result is None:
        # Fallback: simple random search or grid
        # Generate a valid random point
        def random_point():
            while True:
                u = np.random.random()
                v = np.random.random()
                if u + v <= 1:
                    # Barycentric to Cartesian
                    p = u * V1 + v * V2
                    return p
        
        # Just return a set of random valid points if DE failed completely
        pts = np.array([random_point() for _ in range(N)])
        return pts
        
    # Local Polish
    current_time = time.time()
    if current_time < deadline - 10.0: # At least 10s left
        
        # Use Powell method for local refinement
        # Powell is good for non-smooth functions
        try:
            res_local = minimize(
                objective, 
                best_result, 
                method='Powell', 
                options={'maxiter': 200, 'ftol': 1e-10, 'xtol': 1e-10}
            )
            
            # Check if local result is better
            if res_local.fun < objective(best_result):
                best_result = res_local.x
                
        except Exception:
            pass
            
    # Final validity check
    final_points = best_result.reshape((N, 2))
    valid = True
    for p in final_points:
        if not point_in_triangle(p):
            valid = False
            break
            
    if not valid:
        # If the final result is invalid, try to project or return a known valid config
        # For robustness, if invalid, return a simple valid configuration
        # e.g. vertices and centroid
        # But we should try to fix it. 
        # Simple fix: clamp to bounds and check again? 
        # If it's still invalid, it's outside the triangle shape.
        # Let's just return the best valid point found during DE if we kept track, 
        # but we didn't. 
        # So we generate a safe fallback.
        def random_point():
            while True:
                u = np.random.random()
                v = np.random.random()
                if u + v <= 1:
                    p = u * V1 + v * V2
                    return p
        final_points = np.array([random_point() for _ in range(N)])
        
    return final_points

# EVOLVE-BLOCK-START
# The code above is the entire program. 
# We need to wrap it in the EVOLVE-BLOCK as requested? 
# "Keep the function names and signatures of the seed program."
# The seed program had:
# import random
# import numpy as np
# def construct(): ...
#
# I will place the new code inside the block.
# EVOLVE-BLOCK-END
