import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin for safety
    
    best_score = 0.0
    best_points = None
    
    def get_score(points):
        # points: 8x2
        # Scale to diameter 2
        dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        scaled = points * (2.0 / d_max)
        # Product of ordered distances
        # Ordered pairs i != j
        # d_scaled[i, j] for i != j
        # log_prod = sum(log(d_scaled[i, j]) for i != j)
        # This is 2 * sum(log(d_scaled[i, j]) for i < j)
        i, j = np.triu_indices(8, k=1)
        d_scaled = dists[i, j] * (2.0 / d_max)
        d_scaled = np.maximum(d_scaled, 1e-10)
        log_prod_unordered = np.sum(np.log(d_scaled))
        log_prod_ordered = 2.0 * log_prod_unordered
        # Score = exp(log_prod_ordered) / 8^8
        # 8^8 = 2^24
        # log_score = log_prod_ordered - 24 * log(2)
        log_score = log_prod_ordered - 24.0 * np.log(2.0)
        return np.exp(log_score)

    def objective(x):
        # x: 16 floats
        points = x.reshape(8, 2)
        dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        i, j = np.triu_indices(8, k=1)
        d_flat = dists[i, j]
        
        # Soft max for diameter
        tau = 10.0
        d_max_flat = np.max(d_flat)
        # Handle case where all distances are 0 (should not happen with log term)
        if d_max_flat < 1e-10:
            return 1e100
        log_d_soft = d_max_flat + np.log(np.sum(np.exp(tau * (d_flat - d_max_flat)))) / tau
        # Log product
        d_safe = np.maximum(d_flat, 1e-10)
        log_prod = np.sum(np.log(d_safe))
        
        # Maximize log_prod - 28 * log_d_soft
        # Minimize negative
        return -(log_prod - 28.0 * log_d_soft)

    # Initial points: Regular Octagon
    angles = 2 * np.pi * np.arange(8) / 8.0
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    init_flat = init_points.flatten()
    
    # Try multiple random starts and perturbations
    np.random.seed(42)
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            x0 = init_flat
        else:
            # Random start within [-2, 2]
            x0 = np.random.uniform(-2, 2, 16)
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            if res.success or res.fun < 1e100:
                points = res.x.reshape(8, 2)
                score = get_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points
        except Exception as e:
            pass

    # If best_points is None, return regular octagon
    if best_points is None:
        best_points = init_points
    else:
        # Scale best_points to diameter 2
        dists = np.linalg.norm(best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :], axis=2)
        d_max = np.max(dists)
        if d_max > 1e-10:
            best_points = best_points * (2.0 / d_max)

    return [[float(p[0]), float(p[1])] for p in best_points]
