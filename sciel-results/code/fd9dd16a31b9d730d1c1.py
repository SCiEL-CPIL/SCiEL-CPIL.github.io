import numpy as np
from scipy.optimize import differential_evolution
import time

def _log_product(points):
    points = np.array(points)
    n = len(points)
    if n < 2:
        return -np.inf
    
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-10:
                return -np.inf
            dists.append(d)
    
    diameter = max(dists)
    if diameter < 1e-10:
        return -np.inf
    
    scale = 2.0 / diameter
    log_product = np.sum(np.log(dists)) + (n * (n - 1) / 2.0) * np.log(scale)
    return log_product

def _objective(x):
    # x is 16 floats
    points = x.reshape(8, 2)
    # Center points to remove translation invariance
    # This helps the optimizer
    points = points - np.mean(points, axis=0)
    val = _log_product(points)
    if val == -np.inf:
        return -1e10
    return -val

def construct():
    # Bounds for DE
    # We can use a reasonable box, e.g., [-10, 10]
    bounds = [(-10, 10)] * 16
    
    # Regular octagon as a seed? DE doesn't take seeds directly, but we can check it.
    angles = [2 * np.pi * k / 8 for k in range(8)]
    reg_points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    reg_score = _log_product(reg_points)
    
    best_x = None
    best_score = reg_score
    
    # Run Differential Evolution
    # maxiter and popsize control runtime
    # 16 dims, popsize=20, maxiter=100 -> 2000 evals.
    # Each eval is O(n^2) = 64 ops. Very fast.
    # We can afford more.
    start_time = time.time()
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=200, 
            popsize=30, 
            seed=42, 
            tol=1e-5,
            disp=False
        )
        if result.fun < 0: # If it found something better than 0 (which is -log(1))
            curr_score = -result.fun
            if curr_score > best_score:
                best_score = curr_score
                best_x = result.x
    except Exception:
        pass
    
    # If DE didn't find a better solution, try a few random local optimizations from DE's best or random
    # But DE should be good.
    
    if best_x is None:
        return [[float(p[0]), float(p[1])] for p in reg_points]
    
    pts = best_x.reshape(8, 2)
    # Re-center and scale check
    pts = pts - np.mean(pts, axis=0)
    
    # Verify distinctness
    for i in range(8):
        for j in range(i+1, 8):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-5:
                return [[float(p[0]), float(p[1])] for p in reg_points]
                
    return [[float(p[0]), float(p[1])] for p in pts]
