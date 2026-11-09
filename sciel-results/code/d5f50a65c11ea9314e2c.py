"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import numpy as np
import time
import random
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_solution = None
    
    # Number of restarts to attempt. We will check time in the loop.
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # 1. Generate random initial configuration
        # Aspect ratio w/h
        # w + h = 2 (we can fix this equality in the optimizer or just penalize)
        # Let's sample w uniformly from [0.5, 1.5] so h = 2 - w is in [0.5, 1.5]
        w = random.uniform(0.5, 1.5)
        h = 2.0 - w
        
        # Initialize circles
        # x, y, r
        x = np.random.uniform(0.1, w - 0.1, N)
        y = np.random.uniform(0.1, h - 0.1, N)
        r = np.random.uniform(0.05, 0.2, N)
        
        # Ensure initial radii don't violate boundaries too badly
        r = np.minimum(r, np.minimum(x, w - x))
        r = np.minimum(r, np.minimum(y, h - y))
        r = np.maximum(r, 0.01)
        
        # Combine variables: [w, x_0...x_N-1, y_0...y_N-1, r_0...r_N-1]
        # We optimize w, x, y, r. h is derived from w to satisfy perimeter constraint strictly?
        # No, better to optimize w and h separately with penalty, or fix h = 2-w.
        # Fixing h = 2-w reduces dimensionality and ensures perimeter constraint.
        # Variables: w, x[0..N-1], y[0..N-1], r[0..N-1]
        
        def objective(params):
            w_val = params[0]
            h_val = 2.0 - w_val
            
            # If w or h are invalid, return high penalty
            if w_val <= 0.01 or h_val <= 0.01:
                return 1e6
            
            x_val = params[1 : 1+N]
            y_val = params[1+N : 1+2*N]
            r_val = params[1+2*N : 1+3*N]
            
            # Ensure radii are positive
            r_val = np.maximum(r_val, 0.01)
            
            # Boundary penalties
            # x_i must be in [r_i, w-r_i]
            # y_i must be in [r_i, h-r_i]
            
            # If r_i > w/2 or r_i > h/2, it's impossible to fit. Penalty.
            max_r_x = w_val / 2.0
            max_r_y = h_val / 2.0
            
            bound_pen = 0.0
            for i in range(N):
                # x bounds
                if x_val[i] < r_val[i]:
                    bound_pen += (r_val[i] - x_val[i]) ** 2
                elif x_val[i] > w_val - r_val[i]:
                    bound_pen += (x_val[i] - (w_val - r_val[i])) ** 2
                
                # y bounds
                if y_val[i] < r_val[i]:
                    bound_pen += (r_val[i] - y_val[i]) ** 2
                elif y_val[i] > h_val - r_val[i]:
                    bound_pen += (y_val[i] - (h_val - r_val[i])) ** 2
                
                # Radius size limit (cannot be larger than half width/height)
                if r_val[i] > max_r_x:
                    bound_pen += (r_val[i] - max_r_x) ** 2
                if r_val[i] > max_r_y:
                    bound_pen += (r_val[i] - max_r_y) ** 2
            
            # Overlap penalties
            overlap_pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    dx = x_val[i] - x_val[j]
                    dy = y_val[i] - y_val[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = r_val[i] + r_val[j]
                    if dist_sq < min_dist * min_dist:
                        # Smooth penalty: (min_dist - sqrt(dist_sq))^2
                        # To avoid sqrt, we can use (min_dist^2 - dist_sq) / min_dist
                        # Or just (min_dist - sqrt(dist_sq + 1e-9)) ** 2
                        dist = np.sqrt(dist_sq + 1e-9)
                        overlap_pen += (min_dist - dist) ** 2
            
            # Total penalty
            penalty = 100.0 * (bound_pen + overlap_pen)
            
            # Objective: Minimize -sum(r) + penalty
            return -np.sum(r_val) + penalty

        # Initial guess vector
        x0 = np.concatenate(([w], x, y, r))
        
        # Optimization
        # Powell method is robust for non-smooth penalties
        # We limit maxiter to keep it fast
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'maxfev': 1000})
        except Exception:
            continue
            
        # Check time
        if time.time() > deadline:
            break
        
        # Extract best result
        res_w = res.x[0]
        res_h = 2.0 - res_w
        if res_w <= 0.01 or res_h <= 0.01:
            continue
            
        res_x = res.x[1 : 1+N]
        res_y = res.x[1+N : 1+2*N]
        res_r = res.x[1+2*N : 1+3*N]
        res_r = np.maximum(res_r, 0.01)
        
        # Verify validity
        valid = True
        for i in range(N):
            if res_r[i] > res_w/2 or res_r[i] > res_h/2:
                valid = False
                break
            if res_x[i] < res_r[i] - 1e-6 or res_x[i] > res_w - res_r[i] + 1e-6:
                valid = False
                break
            if res_y[i] < res_r[i] - 1e-6 or res_y[i] > res_h - res_r[i] + 1e-6:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dx = res_x[i] - res_x[j]
                dy = res_y[i] - res_y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = res_r[i] + res_r[j]
                if dist_sq < (min_dist - 1e-6)**2:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        score = np.sum(res_r)
        if score > best_score:
            best_score = score
            # Store solution
            # Format: (x, y, r)
            best_solution = np.column_stack((res_x, res_y, res_r))
            
    # If no valid solution found, return a safe fallback (e.g. seed-like grid)
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
# EVOLVE-BLOCK-END
