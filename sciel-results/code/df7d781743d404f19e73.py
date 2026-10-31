import numpy as np
from scipy.optimize import minimize
import time
import random

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TOTAL_AREA = np.sqrt(3) / 4.0

def bary_to_cart(bary_coords):
    u = bary_coords[:, 0]
    v = bary_coords[:, 1]
    w = 1.0 - u - v
    x = v * V1[0] + w * V2[0]
    y = v * V1[1] + w * V2[1]
    return np.column_stack((x, y))

def cart_to_bary(cart_coords):
    w = cart_coords[:, 1] / (np.sqrt(3) / 2)
    v = cart_coords[:, 0] - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack((u, v))

def triangle_areas(points):
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area2 = abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area2 / 2.0)
    return np.array(areas)

def objective_smooth_lbfgs(bary_flat, beta=10.0, penalty_weight=100.0):
    bary = bary_flat.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    pen_u = np.sum(np.maximum(0, -u)**2)
    pen_v = np.sum(np.maximum(0, -v)**2)
    pen_w = np.sum(np.maximum(0, -w)**2)
    penalty = penalty_weight * (pen_u + pen_v + pen_w)
    
    points = bary_to_cart(bary)
    areas = triangle_areas(points)
    
    min_a = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_a))
    log_sum = np.log(np.sum(exp_terms))
    smooth_min_val = min_a - (1.0 / beta) * log_sum
    
    return -smooth_min_val + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_points = None
    best_bary = None
    
    rng = random.Random(42)
    
    bounds = [(0.0, 1.0), (0.0, 1.0)] * N
    
    while time.time() < deadline:
        # Random initialization
        u = np.array([rng.random() for _ in range(N)])
        v = np.array([rng.random() * (1 - u) for _ in range(N)])
        bary_init = np.column_stack((u, v))
        
        # Optimize
        beta = 10.0
        res = minimize(
            objective_smooth_lbfgs, 
            bary_init.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            args=(beta,),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-6}
        )
        
        bary_opt = res.x.reshape(N, 2)
        
        # Clamp to feasible region
        bary_opt[:, 0] = np.clip(bary_opt[:, 0], 0, 1)
        bary_opt[:, 1] = np.clip(bary_opt[:, 1], 0, 1)
        sum_uv = bary_opt[:, 0] + bary_opt[:, 1]
        mask = sum_uv > 1
        if np.any(mask):
            scale = 1.0 / sum_uv[mask]
            bary_opt[mask, 0] *= scale
            bary_opt[mask, 1] *= scale
            
        points = bary_to_cart(bary_opt)
        areas = triangle_areas(points)
        min_area = np.min(areas)
        score = min_area / TOTAL_AREA
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_bary = bary_opt.copy()
            
    # Polish step if time remains
    if best_bary is not None and time.time() < deadline - 5.0:
        # Try higher beta for sharper min
        for beta in [20, 50, 100]:
            if time.time() > deadline - 5.0:
                break
            res = minimize(
                objective_smooth_lbfgs, 
                best_bary.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                args=(beta,),
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            bary_opt = res.x.reshape(N, 2)
            bary_opt[:, 0] = np.clip(bary_opt[:, 0], 0, 1)
            bary_opt[:, 1] = np.clip(bary_opt[:, 1], 0, 1)
            sum_uv = bary_opt[:, 0] + bary_opt[:, 1]
            mask = sum_uv > 1
            if np.any(mask):
                scale = 1.0 / sum_uv[mask]
                bary_opt[mask, 0] *= scale
                bary_opt[mask, 1] *= scale
                
            points = bary_to_cart(bary_opt)
            areas = triangle_areas(points)
            min_area = np.min(areas)
            score = min_area / TOTAL_AREA
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                best_bary = bary_opt.copy()

    if best_points is None:
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        best_points = np.array(points)
        
    return best_points
