import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # Objective function
    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ...]
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Penalty for boundaries
        # x_i should be in [r_i, 1-r_i]
        # y_i should be in [r_i, 1-r_i]
        # r_i should be > 0 (handled by penalty or bounds if possible, but Powell doesn't take bounds easily)
        # Let's penalize r_i < 0 as well
        pen_bound = 0.0
        for i in range(N):
            if r[i] < 0:
                pen_bound += r[i]**2
            if x[i] < r[i]:
                pen_bound += (x[i] - r[i])**2
            if x[i] > 1 - r[i]:
                pen_bound += (x[i] - (1 - r[i]))**2
            if y[i] < r[i]:
                pen_bound += (y[i] - r[i])**2
            if y[i] > 1 - r[i]:
                pen_bin += (y[i] - (1 - r[i]))**2
        
        # Penalty for overlaps
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    pen_overlap += (min_dist - dist)**2
        
        penalty = pen_bound + pen_overlap
        # We want to maximize sum(r), so minimize -sum(r) + lambda * penalty
        # Lambda needs to be large enough to enforce constraints
        lambda_val = 1000.0
        return -np.sum(r) + lambda_val * penalty

    def is_valid(params, tol=1e-6):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        if np.any(r < 0):
            return False
        for i in range(N):
            if x[i] < r[i] - tol or x[i] > 1 - r[i] + tol:
                return False
            if y[i] < r[i] - tol or y[i] > 1 - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    best_score = -np.inf
    best_params = None
    
    # Multi-start
    num_restarts = 15
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Random positions in [0.1, 0.9]
        x0 = np.random.uniform(0.1, 0.9, N)
        y0 = np.random.uniform(0.1, 0.9, N)
        # Small initial radii
        r0 = np.full(N, 0.05)
        
        # Interleave
        params0 = np.zeros(3*N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        # Optimize
        try:
            res = minimize(objective, params0, method='Powell', 
                           options={'maxiter': 1000, 'maxfev': 10000, 'disp': False})
            
            if res.success or is_valid(res.x):
                score = -res.fun - 1000.0 * 0 # Wait, res.fun includes penalty.
                # We need the raw score sum(r) if valid.
                # If valid, penalty is 0, so res.fun = -sum(r).
                # So sum(r) = -res.fun.
                # But if penalty is not 0, res.fun is misleading for score.
                # We check validity first.
                if is_valid(res.x):
                    sum_r = np.sum(res.x[2::3])
                    if sum_r > best_score:
                        best_score = sum_r
                        best_params = res.x.copy()
        except Exception as e:
            pass

    # If no valid solution found, return a trivial one (grid)
    if best_params is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Polish: Try to refine the best solution found
    if time.time() < deadline - 10:
        try:
            res_polish = minimize(objective, best_params, method='Powell', 
                                  options={'maxiter': 500, 'maxfev': 5000, 'disp': False})
            if is_valid(res_polish.x):
                sum_r = np.sum(res_polish.x[2::3])
                if sum_r > best_score:
                    best_score = sum_r
                    best_params = res_polish.x.copy()
        except Exception:
            pass

    # Construct final array
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    
    result = np.column_stack((x, y, r))
    return result
