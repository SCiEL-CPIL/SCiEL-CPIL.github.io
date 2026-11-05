import numpy as np
from scipy.optimize import minimize
import random

def _compute_score(points):
    points = np.array(points)
    n = len(points)
    if n < 2:
        return 0.0
    
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

def _negative_score(points_flat):
    n = 8
    points = np.array(points_flat).reshape(n, 2)
    score = _compute_score(points)
    if score == -np.inf:
        return 1e10
    return -score

def construct():
    best_score = -np.inf
    best_points = None
    
    # Regular octagon
    angles = [2 * np.pi * k / 8 for k in range(8)]
    reg_points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    reg_score = _compute_score(reg_points)
    best_score = reg_score
    best_points = reg_points.flatten()
    
    # Try a few random starts with local optimization
    random.seed(42)
    for _ in range(20):
        # Random points in a box
        points = np.random.uniform(-1, 1, (8, 2))
        # Ensure they are distinct enough
        x0 = points.flatten()
        
        try:
            result = minimize(_negative_score, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.success:
                curr_score = -result.fun
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = result.x
        except Exception:
            pass
            
    # Also try perturbing the regular octagon
    for _ in range(10):
        points = reg_points + np.random.normal(0, 0.1, (8, 2))
        x0 = points.flatten()
        try:
            result = minimize(_negative_score, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.success:
                curr_score = -result.fun
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = result.x
        except Exception:
            pass

    if best_points is not None:
        pts = best_points.reshape(8, 2)
        # Check validity
        for i in range(8):
            for j in range(i+1, 8):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-5:
                    return [[float(np.cos(2 * np.pi * k / 8)), float(np.sin(2 * np.pi * k / 8))] for k in range(8)]
        return [[float(p[0]), float(p[1])] for p in pts]
    
    return [[float(np.cos(2 * np.pi * k / 8)), float(np.sin(2 * np.pi * k / 8))] for k in range(8)]
