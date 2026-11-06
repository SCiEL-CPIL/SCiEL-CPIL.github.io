import numpy as np
from scipy.optimize import minimize
from math import cos, sin, pi, log
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    n = 8
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_count = len(i_idx)
    
    def get_points(x):
        return x.reshape(n, 2)
    
    def distances(points):
        # points is (n, 2)
        # Compute all pairwise distances
        # diff[i, j] = points[i] - points[j]
        # We only need i < j
        diffs = points[i_idx] - points[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        return dists
    
    def log_product_of_distances(x):
        points = get_points(x)
        dists = distances(points)
        # Avoid log(0)
        dists = np.maximum(dists, 1e-12)
        return np.sum(np.log(dists))
    
    def diameter(x):
        points = get_points(x)
        dists = distances(points)
        return np.max(dists)
    
    def objective_with_penalty(x, penalty_weight, tau):
        points = get_points(x)
        dists = distances(points)
        
        # Log product term
        safe_dists = np.maximum(dists, 1e-12)
        log_prod = np.sum(np.log(safe_dists))
        
        # Soft diameter term
        # D_soft = tau * log(sum(exp(d/tau)))
        # We want to penalize if D_soft > 2
        # Using a quadratic penalty on the violation
        # To avoid overflow in exp, shift by max
        max_d = np.max(dists)
        if max_d > 0:
            exp_terms = np.exp((dists - max_d) / tau)
            d_soft = max_d + tau * np.log(np.sum(exp_terms))
        else:
            d_soft = 0.0
            
        violation = max(0.0, d_soft - 2.0)
        penalty = penalty_weight * violation * violation
        
        # We want to maximize log_prod, so minimize -log_prod + penalty
        return -log_prod + penalty
    
    def run_optimization(x0, penalty_weights, tau_values):
        current_x = x0.copy()
        best_x = x0.copy()
        best_score = -np.inf
        
        for pw, tau in zip(penalty_weights, tau_values):
            if time.time() > deadline:
                break
            
            # Define the function to minimize
            func = lambda x: objective_with_penalty(x, pw, tau)
            
            try:
                res = minimize(func, current_x, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
                if res.success:
                    current_x = res.x
            except Exception:
                pass
            
            # Evaluate the true constrained score for this candidate
            # Score is based on scaled diameter 2.
            # If diameter <= 2, score is proportional to product.
            # If diameter > 2, we should have scaled down, but the penalty should have prevented this.
            # Let's compute the actual score that the evaluator would give.
            points = get_points(current_x)
            dists = distances(points)
            d_max = np.max(dists)
            
            if d_max < 1e-9:
                continue
                
            # The evaluator scales to diameter 2.
            # scale = 2 / d_max
            # scaled_dists = dists * scale
            # log_score = sum(log(scaled_dists)) - 8*log(8)
            # log_score = sum(log(dists)) + 28*log(2) - 28*log(d_max) - 8*log(8)
            
            log_prod = np.sum(np.log(np.maximum(dists, 1e-12)))
            log_d_max = np.log(d_max)
            
            # Constant terms for comparison:
            # We just need to compare the relative value.
            # Score = (2^28 / d_max^28) * prod(dists) / 8^8
            # log(Score) = 28*log(2) - 28*log(d_max) + log_prod - 8*log(8)
            
            log_score_val = 28.0 * np.log(2.0) - 28.0 * log_d_max + log_prod - 8.0 * np.log(8.0)
            
            if log_score_val > best_score:
                best_score = log_score_val
                best_x = current_x.copy()
                
        return best_x, best_score

    # Initial configurations
    # 1. Regular octagon
    x_octagon = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]).flatten()
    
    # 2. Random initializations
    np.random.seed(42)
    initial_points = [x_octagon]
    for _ in range(10):
        # Random points in a circle of radius 1
        r = np.sqrt(np.random.rand(n))
        theta = np.random.rand(n) * 2 * pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initial_points.append(pts.flatten())
        
    # Add some structured perturbations of the octagon
    for _ in range(5):
        pert = np.random.normal(0, 0.1, size=16)
        initial_points.append(x_octagon + pert)

    best_x_global = x_octagon
    best_score_global = -np.inf
    
    # Optimization schedule
    # Increase penalty weight to enforce constraint
    penalty_weights = [10, 100, 1000, 10000, 100000]
    tau_values = [0.5, 0.2, 0.1, 0.05, 0.01]
    
    for x0 in initial_points:
        if time.time() > deadline:
            break
            
        x_res, score_res = run_optimization(x0, penalty_weights, tau_values)
        
        if score_res > best_score_global:
            best_score_global = score_res
            best_x_global = x_res

    # Final check and formatting
    points = get_points(best_x_global)
    
    # Ensure coordinates are within limits
    # The problem says |coordinate| < 10^6. Our points are around 1, so this is fine.
    
    # Return list of lists
    return [[float(x), float(y)] for x, y in points]

if __name__ == "__main__":
    construct()
