import numpy as np
import time
import sys
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    timeout = 600.0  # Safety margin before 630s
    
    best_score = -np.inf
    best_solution = None
    
    # Baseline: 3x3 grid
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    baseline_score = np.sum(sides)
    
    if baseline_score > best_score:
        best_score = baseline_score
        best_solution = (centers.copy(), sides.copy())
        
    # Objective function: Minimize -sum(sides) + penalty
    def objective(params):
        # params shape: (30,) -> [cx1, cy1, s1, cx2, cy2, s2, ...]
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        # Ensure sides are positive
        s = np.maximum(s, 1e-6)
        
        # Boundary penalty
        # x bounds: cx - s/2 >= 0 => s/2 - cx <= 0
        #           cx + s/2 <= 1 => cx + s/2 - 1 >= 0
        # We use squared penalties for smoothness
        p_bound = 0.0
        for i in range(n):
            p_bound += np.max([0.0, (s[i]/2.0 - cx[i])])**2
            p_bound += np.max([0.0, (cx[i] + s[i]/2.0 - 1.0)])**2
            p_bound += np.max([0.0, (s[i]/2.0 - cy[i])])**2
            p_bound += np.max([0.0, (cy[i] + s[i]/2.0 - 1.0)])**2
            
        # Overlap penalty
        p_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # If they overlap, the "penetration" in x and y
                # We want to penalize if dx < (si+sj)/2 AND dy < (si+sj)/2
                # A simple repulsive force:
                pen_x = np.max([0.0, (s[i] + s[j])/2.0 - dx])
                pen_y = np.max([0.0, (s[i] + s[j])/2.0 - dy])
                # If both are positive, they overlap.
                # Sum of penetrations is a good metric
                p_overlap += pen_x + pen_y
                
        penalty_weight = 100.0
        return -np.sum(s) + penalty_weight * (p_bound + p_overlap)

    # Run multiple optimizations with different initial guesses
    np.random.seed(42)
    
    max_iterations = 20 # Number of restarts
    for i in range(max_iterations):
        if time.time() - start_time > timeout - 60:
            break
            
        # Initialize
        if i == 0:
            # Use baseline
            init_cx = centers.copy()
            init_cy = centers.copy()[:, 1]
            init_s = sides.copy()
        else:
            # Random perturbation of baseline or random start
            if i % 2 == 0:
                # Perturb baseline
                init_cx = centers[:, 0] + np.random.normal(0, 0.05, n)
                init_cy = centers[:, 1] + np.random.normal(0, 0.05, n)
                init_s = sides * (1 + np.random.normal(0, 0.1, n))
            else:
                # Random valid-ish start
                init_s = np.random.uniform(0.1, 0.4, n)
                init_cx = np.random.uniform(0.1, 0.9, n)
                init_cy = np.random.uniform(0.1, 0.9, n)
                
        # Flatten
        init_params = np.zeros(3*n)
        init_params[0::3] = init_cx
        init_params[1::3] = init_cy
        init_params[2::3] = init_s
        
        # Bounds
        # cx, cy in [0, 1]
        # s in [0.01, 1.0]
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.0)) # cx
            bounds.append((0.0, 1.0)) # cy
            bounds.append((0.01, 1.0)) # s
            
        try:
            res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success:
                # Extract solution
                res_cx = res.x[0::3]
                res_cy = res.x[1::3]
                res_s = res.x[2::3]
                
                # Validate strictly
                valid = True
                for idx in range(n):
                    if res_s[idx] <= 0:
                        valid = False
                        break
                    if res_cx[idx] - res_s[idx]/2 < -1e-7 or res_cx[idx] + res_s[idx]/2 > 1 + 1e-7:
                        valid = False
                        break
                    if res_cy[idx] - res_s[idx]/2 < -1e-7 or res_cy[idx] + res_s[idx]/2 > 1 + 1e-7:
                        valid = False
                        break
                        
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            t = (res_s[i] + res_s[j])/2.0 - 1e-7
                            if abs(res_cx[i] - res_cx[j]) < t and abs(res_cy[i] - res_cy[j]) < t:
                                valid = False
                                break
                        if not valid:
                            break
                            
                if valid:
                    score = np.sum(res_s)
                    if score > best_score:
                        best_score = score
                        best_solution = (np.stack([res_cx, res_cy], axis=1), res_s)
                        
        except Exception as e:
            pass

    return best_solution

