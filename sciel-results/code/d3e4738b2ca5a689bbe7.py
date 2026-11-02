import numpy as np
import random
import time
import math
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2

# Precompute triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

def _vectorized_min_area_cart(points):
    p_i = points[_T_I]
    p_j = points[_T_J]
    p_k = points[_T_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _soft_min_area_cart(points, tau):
    p_i = points[_T_I]
    p_j = points[_T_J]
    p_k = points[_T_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    max_a = np.max(areas)
    exps = np.exp(-(areas - max_a) / tau)
    sum_exps = np.sum(exps)
    soft_min = max_a - tau * np.log(sum_exps)
    return soft_min

def _penalty(points):
    w = 2.0 * points[:, 1] / S3
    v = points[:, 0] - points[:, 1] / S3
    u = 1.0 - points[:, 0] - points[:, 1] / S3
    
    pen = 0.0
    pen += np.sum(np.minimum(u, 0)**2)
    pen += np.sum(np.minimum(v, 0)**2)
    pen += np.sum(np.minimum(w, 0)**2)
    
    return pen

def _objective(x, tau):
    pts = x.reshape(N, 2)
    val = _soft_min_area_cart(pts, tau)
    pen = _penalty(pts)
    return -val + pen * 100.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def random_point():
        while True:
            u = random.random()
            v = random.random()
            if u + v < 1.0:
                x = u * 1.0 + v * 0.5
                y = u * 0.0 + v * S3H
                return np.array([x, y])

    best_points = None
    best_val = -1.0
    
    num_restarts = 15
    steps_per_restart = 30000
    
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        points = np.array([random_point() for _ in range(N)])
        
        T = 0.01
        current_val = _vectorized_min_area_cart(points)
        current_points = points.copy()
        step_size = 0.05
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 15.0:
                break
                
            idx = random.randint(0, N-1)
            old_p = current_points[idx].copy()
            
            noise = np.random.normal(0, step_size, 2)
            new_p = old_p + noise
            
            w = 2.0 * new_p[1] / S3
            v = new_p[0] - new_p[1] / S3
            u = 1.0 - new_p[0] - new_p[1] / S3
            
            if u < 0 or v < 0 or w < 0:
                continue
                
            current_points[idx] = new_p
            new_val = _vectorized_min_area_cart(current_points)
            
            delta = new_val - current_val
            if delta > 0 or random.random() < math.exp(delta / T):
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_points = current_points.copy()
            else:
                current_points[idx] = old_p
                
            T *= 0.9999
            if step % 10000 == 0 and step > 0:
                T = 0.005

    if best_points is not None and time.time() < deadline - 5.0:
        tau = 0.001
        x0 = best_points.flatten()
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', args=(tau,),
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < -best_val + 1e-9:
                final_pts = res.x.reshape(N, 2)
                pen = _penalty(final_pts)
                if pen < 1e-6:
                    final_val = _vectorized_min_area_cart(final_pts)
                    if final_val > best_val:
                        best_points = final_pts
                        best_val = final_val
        except Exception:
            pass

    if best_points is None:
        best_points = np.array([random_point() for _ in range(N)])
        
    return best_points
