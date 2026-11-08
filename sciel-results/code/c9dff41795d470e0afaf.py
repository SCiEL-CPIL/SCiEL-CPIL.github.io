import numpy as np
import time
from scipy.optimize import differential_evolution
import random

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0
H = np.sqrt(3) / 2.0

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 of the points.
    points: (n, 2)
    Returns normalized min area (divided by total triangle area).
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized computation of all triangle areas
    # We want area for all triples (i, j, k) with i < j < k
    # Area = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    
    # For n=11, there are 165 triangles.
    # We can create indices for all triples.
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
                
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    # Extract coordinates
    xi = points[i_idx, 0]
    yi = points[i_idx, 1]
    xj = points[j_idx, 0]
    yj = points[j_idx, 1]
    xk = points[k_idx, 0]
    yk = points[k_idx, 1]
    
    # Compute 2x area (double area to avoid 0.5 factor)
    # 2*Area = |(xj - xi)(yk - yi) - (yj - yi)(xk - xi)|
    double_areas = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    
    min_double_area = np.min(double_areas)
    min_area = 0.5 * min_double_area
    
    return min_area / AREA

def _is_inside_triangle(points):
    """
    Check if points are inside the triangle.
    Returns a boolean mask.
    """
    # points: (n, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    mask = y >= -1e-9
    
    # x >= 0
    mask &= x >= -1e-9
    
    # x <= 1
    mask &= x <= 1.0 + 1e-9
    
    # y <= sqrt(3) * x  (left side)
    mask &= y <= np.sqrt(3) * x + 1e-9
    
    # y <= sqrt(3) * (1 - x) (right side)
    mask &= y <= np.sqrt(3) * (1.0 - x) + 1e-9
    
    return mask

def _objective(params):
    """
    Objective function for minimization.
    params: (22,) array of x, y coordinates for 11 points.
    Returns negative of min area (to maximize) + penalty if outside.
    """
    points = params.reshape(11, 2)
    
    # Check bounds
    if not np.all(_is_inside_triangle(points)):
        # Penalty: distance outside? Or just large constant?
        # A large constant works well for DE if the penalty is significant.
        # Let's compute a penalty based on violation magnitude to guide back.
        x = points[:, 0]
        y = points[:, 1]
        
        # Violations
        v1 = np.maximum(0, -y)
        v2 = np.maximum(0, -x)
        v3 = np.maximum(0, x - 1.0)
        v4 = np.maximum(0, y - np.sqrt(3) * x)
        v5 = np.maximum(0, y - np.sqrt(3) * (1.0 - x))
        
        penalty = np.sum(v1**2 + v2**2 + v3**2 + v4**2 + v5**2)
        return 100.0 + penalty
        
    min_area = _min_triangle_area(points)
    return -min_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Bounds for each point: x in [0, 1], y in [0, H]
    # This is a loose bound, the objective function handles the triangular constraint via penalty.
    # However, tighter bounds help DE.
    # Since points are independent in the vector representation, we can use global bounds.
    # x: [0, 1], y: [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * 11
    
    # Differential Evolution settings
    # We want to explore the global space.
    # popsize: number of individuals in population. 15 * 22 = 330 individuals.
    # This is quite large but n=11 is small so evaluation is fast.
    # Let's estimate time per evaluation.
    # _min_triangle_area does ~165 ops. Vectorized, it's very fast.
    # Let's try popsize=10 to start, maybe increase if time allows.
    
    # Strategy: Run DE with a time limit.
    # DE in SciPy doesn't have a built-in time limit, only maxiter.
    # We can wrap the objective to check time, but that stops the whole run.
    # Better: Estimate maxiter.
    
    # Let's try a few configurations or just one robust one.
    # Given the non-convexity, DE is good.
    
    # We can also try multiple restarts with smaller populations if one DE run is too slow.
    # But DE is global, so one good run should suffice.
    
    # Let's use a custom callback to stop if time is running out?
    # No, scipy DE doesn't support early stopping via callback easily without breaking.
    # We will set maxiter conservatively and check time after.
    
    # Let's profile roughly.
    # One eval: ~100 us?
    # Popsize 10 * 22 = 220 individuals.
    # 220 * 100 us = 22 ms per generation.
    # 100 generations = 2.2 s.
    # We have 140 s. So we can afford many generations.
    # Let's set maxiter=500 and popsize=15.
    
    best_result = None
    best_val = -np.inf
    
    # Run Differential Evolution
    try:
        # seed=42 for reproducibility
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=500, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=True # Use L-BFGS-B to polish the result
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_result = res.x
            
    except Exception as e:
        # If DE fails, try a fallback
        pass

    # If DE didn't find a valid solution or timed out (it shouldn't, but just in case)
    if best_result is None or time.time() > deadline:
        # Fallback: Random sampling with local improvement?
        # Or just return a random valid point if we have no better.
        # But DE should work.
        
        # If we are close to deadline, return whatever we have.
        if best_result is not None:
             points = best_result.reshape(11, 2)
             # Ensure validity
             if np.all(_is_inside_triangle(points)):
                 return points

    # Polish the result with a local optimizer if time permits
    if best_result is not None and time.time() < deadline - 10:
        from scipy.optimize import minimize
        
        # Use L-BFGS-B with bounds
        # We need to map the triangular constraint.
        # The penalty method in _objective is differentiable almost everywhere.
        # L-BFGS-B might struggle with the kinks in the penalty/area function.
        # But `polish=True` in DE already does this.
        
        # Let's do a final check and return.
        points = best_result.reshape(11, 2)
        
        # Verify validity
        if not np.all(_is_inside_triangle(points)):
             # If somehow invalid, try to project or use a random valid one
             # For now, if DE returns invalid, it's a problem.
             # But penalty should keep it valid.
             pass

        return points

    # Fallback if nothing worked
    # Generate 11 random points inside triangle
    rng = random.Random(42)
    points = []
    for _ in range(11):
        while True:
            x = rng.uniform(0, 1)
            y = rng.uniform(0, H)
            if y <= np.sqrt(3) * x and y <= np.sqrt(3) * (1 - x):
                points.append([x, y])
                break
    return np.array(points)

if __name__ == "__main__":
    pts = construct()
    print("Score:", _min_triangle_area(pts))
