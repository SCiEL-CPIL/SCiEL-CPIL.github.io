import numpy as np
from scipy.optimize import minimize
import math

def construct():
    def objective(theta):
        # theta: 4 angles
        angles = np.concatenate([theta, theta + np.pi])
        
        # Pairwise distances
        diffs = np.subtract.outer(angles, angles)
        mask = ~np.eye(8, dtype=bool)
        diffs = diffs[mask]
        
        dists = 2.0 * np.abs(np.sin(diffs / 2.0))
        D = np.max(dists)
        
        if D < 1e-12:
            return 0.0
            
        # Sum of logs
        # Use log1p or just log with clipping
        # dists are positive
        log_dists = np.log(dists)
        sum_log_dists = np.sum(log_dists)
        
        val = 56.0 * math.log(2.0) - 56.0 * math.log(D) + sum_log_dists
        return -val

    # Regular octagon angles for the 4 unique pairs
    # 0, 45, 90, 135 degrees
    theta_reg = np.array([0.0, np.pi/4, np.pi/2, 3*np.pi/4])
    
    best_val = -np.inf
    best_theta = theta_reg
    
    rng = np.random.RandomState(123)
    
    # Try the regular octagon first
    res = minimize(objective, theta_reg, method='Powell', options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12})
    if res.fun < best_val:
        best_val = res.fun
        best_theta = res.x
        
    # Try random perturbations
    for i in range(30):
        # Small perturbation around regular octagon
        if i < 10:
            start = theta_reg + rng.uniform(-0.1, 0.1, 4)
        else:
            # Random start
            start = rng.uniform(0, 2*np.pi, 4)
            
        res = minimize(objective, start, method='Powell', options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_theta = res.x
            
    # Construct points
    angles = np.concatenate([best_theta, best_theta + np.pi])
    points = [[float(np.cos(a)), float(np.sin(a))] for a in angles]
    
    return points
