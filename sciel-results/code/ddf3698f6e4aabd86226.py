import numpy as np
from scipy.optimize import differential_evolution
import math

N = 11
S3 = math.sqrt(3)

def _min_area_vec(points):
    """
    Compute the minimum area of any triangle formed by three of the given points.
    points: (N, 2) array
    """
    min_area = float('inf')
    n = len(points)
    # Using a slightly optimized loop
    for i in range(n):
        p1 = points[i]
        for j in range(i + 1, n):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    return min_area

def _objective_de(params):
    """
    Objective function for Differential Evolution (minimize).
    Returns -min_area + penalty.
    """
    points = params.reshape(N, 2)
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for i in range(N):
        x, y = points[i]
        # Triangle inequalities for y >= 0, y <= S3*x, y <= S3*(1-x)
        if y < 0:
            penalty += (0 - y) ** 2
        if y > S3 * x:
            penalty += (y - S3 * x) ** 2
        if y > S3 * (1 - x):
            penalty += (y - S3 * (1 - x)) ** 2
            
    if penalty > 1e-6:
        return 1000.0 + penalty

    min_area = _min_area_vec(points)
    return -min_area

def construct():
    """
    Place 11 points in the unit equilateral triangle to maximize the minimum triangle area.
    Uses Differential Evolution with Sobol initialization and Nelder-Mead polish.
    """
    # Bounds for each coordinate
    # x in [0, 1], y in [0, S3/2]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, S3 / 2.0))
        
    try:
        result = differential_evolution(
            _objective_de, 
            bounds, 
            maxiter=60, 
            popsize=12, 
            tol=1e-8, 
            seed=42,
            polish=True, 
            init='sobol'
        )
        
        if result.fun < 500:
            points = result.x.reshape(N, 2)
            # Verify validity
            valid = True
            for i in range(N):
                x, y = points[i]
                if y < -1e-9 or y > S3 * x + 1e-9 or y > S3 * (1 - x) + 1e-9:
                    valid = False
                    break
            if valid:
                return points
        else:
            # If the best result is invalid (penalty), we might need to fallback
            # But DE usually finds valid points quickly due to penalty.
            pass
            
    except Exception:
        pass

    # Fallback: Multi-start Nelder-Mead (original approach) if DE fails or is too slow/invalid
    # This ensures we always return something valid.
    best_points = None
    best_score = -float('inf')
    np.random.seed(42)
    
    for _ in range(20): # Reduced restarts for speed
        points = np.zeros((N, 2))
        for i in range(N):
            x = np.random.uniform(0, 1)
            y_max = S3 * min(x, 1 - x)
            y = np.random.uniform(0, y_max)
            points[i] = [x, y]
            
        # Quick local optimization
        # Using a simple gradient-free step or just evaluating
        # For safety, we just return the best random point if DE failed completely?
        # No, let's do a few Nelder-Mead steps.
        from scipy.optimize import minimize
        res = minimize(
            _objective_de,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res.fun < 500:
            score = -res.fun
            if score > best_score:
                best_score = score
                best_points = res.x.reshape(N, 2)
                
    if best_points is not None:
        return best_points
        
    # Final Fallback: Centroid
    return np.full((N, 2), [0.5, S3/6])
