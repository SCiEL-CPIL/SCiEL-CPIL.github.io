import numpy as np
from scipy.optimize import differential_evolution
import time

N = 11

# Pre-compute triangle indices
from itertools import combinations
tri_indices = np.array(list(combinations(range(N), 3)))

def objective(params):
    # params is a flat array of shape (22,)
    points = params.reshape(N, 2)
    x = points[:, 0]
    y = points[:, 1]
    
    # Check if points are inside the triangle
    # Vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Inequalities:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  => sqrt(3)*x - y >= 0
    # 3. y <= -sqrt(3)*(x-1) => -sqrt(3)*x + y + sqrt(3) >= 0
    
    s3 = np.sqrt(3)
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, -(s3 * x - y))
    v3 = np.maximum(0, -(-s3 * x + y + s3))
    
    penalty = np.sum(v1**2 + v2**2 + v3**2) * 100.0
    
    if penalty > 1e-6:
        # If significantly outside, return large penalty
        # This helps DE push points back in
        return penalty
    
    # Calculate areas
    p1 = points[tri_indices[:, 0]]
    p2 = points[tri_indices[:, 1]]
    p3 = points[tri_indices[:, 2]]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return -min_area

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Bounds for x and y
    # x in [0, 1], y in [0, sqrt(3)/2]
    h = np.sqrt(3) / 2
    bounds = [(0, 1), (0, h)] * N
    
    # We will run differential evolution
    # To improve chances, we can run multiple restarts if time allows
    best_result = None
    best_score = -np.inf
    
    # Use a fixed seed for reproducibility in the first run, then vary
    seeds = [42, 123, 456, 789, 101112]
    
    for seed in seeds:
        if time.time() - start_time > timeout:
            break
            
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=seed, 
                maxiter=200, 
                popsize=15, 
                tol=1e-6,
                polish=True
            )
            
            if res.success or res.fun < best_score:
                # Check if the best point is actually inside
                best_params = res.x
                best_points = best_params.reshape(N, 2)
                # Re-verify validity
                x = best_points[:, 0]
                y = best_points[:, 1]
                s3 = np.sqrt(3)
                v1 = np.any(y < -1e-9)
                v2 = np.any(s3 * x - y < -1e-9)
                v3 = np.any(-s3 * x + y + s3 < -1e-9)
                
                if not (v1 or v2 or v3):
                    # Calculate actual score for comparison
                    p1 = best_points[tri_indices[:, 0]]
                    p2 = best_points[tri_indices[:, 1]]
                    p3 = best_points[tri_indices[:, 2]]
                    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
                    areas = 0.5 * np.abs(cross)
                    score = np.min(areas)
                    
                    if score > best_score:
                        best_score = score
                        best_result = best_points.copy()
                        
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to random if DE failed completely
        rng = np.random.RandomState(0)
        pts = []
        for _ in range(N):
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                pts.append([u + 0.5*v, h*v])
            else:
                pts.append([0.5, h/3]) # Center
        best_result = np.array(pts)
        
    return best_result
