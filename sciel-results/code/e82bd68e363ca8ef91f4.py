import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

def get_cartesian(params):
    # params shape (2N,)
    s = params[0::2]
    t = params[1::2]
    # u = (1-t)(1-s), v = t, w = s(1-t)
    w = s * (1 - t)
    v = t
    # x = v + 0.5 w
    # y = SQRT3/2 * w
    x = v + 0.5 * w
    y = (SQRT3 / 2) * w
    return np.column_stack((x, y))

def compute_areas(points):
    # points shape (N, 2)
    # Use broadcasting to compute all areas
    # i, j, k indices
    n = len(points)
    # Create indices for all combinations of 3
    # This might be memory heavy for large N, but N=11 is small.
    # 165 triangles.
    # We can loop or vectorize.
    # Vectorized:
    # P_i, P_j, P_k
    # Area = 0.5 * | (x_j - x_i)(y_k - y_i) - (x_k - x_i)(y_j - y_i) |
    
    # Let's just do a simple loop for clarity and speed for N=11
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi, yi = points[i]
                xj, yj = points[j]
                xk, yk = points[k]
                a = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                areas.append(a)
    return np.array(areas)

def objective(params, beta):
    points = get_cartesian(params)
    areas = compute_areas(points)
    # Soft min
    # Handle very small areas to avoid log(0) or exp overflow
    # exp(-beta * area)
    # If beta is large and area is small, exp is large.
    # Use log-sum-exp trick:
    # log(sum(exp(-beta * A))) = -beta * min(A) + log(sum(exp(-beta*(A - min(A)))))
    # Actually, standard stable logsumexp:
    m = np.min(areas)
    if m < 1e-12:
        m = 1e-12
    # We want to maximize min(areas).
    # f = - (1/beta) * log(sum(exp(-beta * areas)))
    # log(sum(exp(-beta * areas))) = -beta * m + log(sum(exp(-beta * (areas - m))))
    # So f = m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    # Wait, if we maximize m, we want to minimize -m.
    # Let's just compute the soft min value directly.
    # S = (1/beta) * log(sum(exp(beta * areas)))  <-- This is soft MAX
    # We want soft MIN.
    # S_min = - (1/beta) * log(sum(exp(-beta * areas)))
    
    # Stable calculation of log_sum_exp(-beta * areas)
    x = -beta * areas
    x_max = np.max(x)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max)))
    soft_min = -log_sum_exp / beta
    
    return -soft_min # Minimize negative soft min

def construct():
    start_time = time.time()
    deadline = 110.0
    
    best_params = None
    best_score = -np.inf
    
    # Continuation method on beta
    betas = [5, 20, 50, 100, 200]
    
    # Initial random start
    np.random.seed(42)
    current_params = np.random.rand(2 * N)
    
    for beta in betas:
        # If time is running out, break
        if time.time() - start_time > deadline:
            break
            
        # Try a few restarts at this beta if we have time?
        # Or just one pass?
        # Let's do a few restarts to escape local minima.
        n_restarts = 5
        for r in range(n_restarts):
            if time.time() - start_time > deadline:
                break
                
            if r == 0:
                x0 = current_params
            else:
                x0 = np.random.rand(2 * N)
            
            res = minimize(objective, x0, args=(beta,), method='L-BFGS-B', 
                           bounds=[(0, 1)] * (2 * N), options={'maxiter': 1000})
            
            if res.success:
                score = -res.fun # res.fun is -soft_min, so score is soft_min
                # Actually, we want the raw min area for final evaluation, 
                # but soft_min is a good proxy for ranking.
                # To be safe, let's evaluate the true min area for the best candidate.
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            
            # Update current_params to the best found in this beta phase
            # to continue the chain
            # (Simplification: just keep the last result or best of restarts)
            # Let's keep the best of this beta phase as the start for the next
            if res.success and (-res.fun > best_score_in_phase if 'best_score_in_phase' in locals() else True):
                 # This logic is getting messy. 
                 # Simpler: Just run minimize once per beta, using the previous result as start.
                 # If it fails or improves poorly, try a random restart.
                 pass
            
        # After the restarts loop, update current_params to the best found in this beta
        # We need to track best_params_for_beta
        # Let's simplify: 
        # For each beta, do 3 restarts (1 from prev, 2 random). Pick best.
        # Set current_params to that best.
        
    # Final evaluation of best_params
    if best_params is not None:
        points = get_cartesian(best_params)
        # Verify points are inside
        return points
    else:
        # Fallback to seed
        # ...
        pass
