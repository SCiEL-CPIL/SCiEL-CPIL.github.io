"""Refined Cohn-Goncalves Laguerre-polynomial construction for Fourier sign uncertainty."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 600s limit
    
    m = 12
    n = 1
    
    # Seed: arithmetic progression 4, 8, ..., 48
    x0 = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Define the objective: upper bound on C4
    # For Cohn-Goncalves construction, the bound is typically related to the 
    # largest root or a weighted sum. We use the largest root as a proxy,
    # but also consider the harmonic structure.
    # A more accurate bound: C = max(r_i) * (1 + 1/m) or similar.
    # Let's use the largest root as the primary objective.
    
    def objective(x):
        # x must be positive and sorted
        if np.any(x <= 0):
            return 1e10
        # Bound: use the largest root scaled by a factor from Laguerre theory
        # For m double roots, the sign uncertainty bound ~ 2*m, but C4 is a constant.
        # The actual bound in Cohn-Goncalves is the point where the constructed
        # function's last sign change occurs, which is the largest root.
        return np.max(x)
    
    # Constraints: roots must be positive and strictly increasing
    constraints = []
    # Add constraints that x is sorted and positive
    # Use inequality constraints: x[i+1] > x[i] for all i
    for i in range(m - 1):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i+1] - x[i] - 0.01
        })
    # Positivity
    for i in range(m):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[i] - 0.01
        })
    
    bounds = [(0.1, 100.0) for _ in range(m)]
    
    best_x = x0.copy()
    best_score = objective(x0)
    
    # Try multiple optimization methods
    methods = ['SLSQP', 'trust-constr', 'L-BFGS-B']
    
    for method in methods:
        if time.time() > deadline - 30:
            break
        
        try:
            if method == 'L-BFGS-B':
                # L-BFGS-B doesn't handle inequality constraints well, 
                # so we'll use a penalty approach
                def penalized_obj(x):
                    if np.any(x <= 0.01):
                        return 1e10
                    sorted_x = np.sort(x)
                    if np.any(np.diff(sorted_x) < 0.01):
                        return 1e10
                    return np.max(sorted_x) + 0.1 * np.sum(1.0 / sorted_x)
                
                res = minimize(penalized_obj, x0, method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-12})
                if res.success and res.fun < best_score:
                    # Sort the result
                    best_x = np.sort(res.x)
                    best_score = np.max(best_x)
            else:
                res = minimize(objective, x0, method=method,
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 5000, 'ftol': 1e-12})
                if res.success and res.fun < best_score:
                    best_x = np.sort(res.x)
                    best_score = np.max(best_x)
        except Exception:
            continue
    
    # Also try a geometric progression starting from seed
    if time.time() < deadline - 30:
        # Try logarithmic spacing
        for log_base in [1.05, 1.1, 1.15, 1.2]:
            candidate = 4.0 * np.array([log_base ** i for i in range(m)])
            if candidate[-1] < 100:
                score = np.max(candidate)
                if score < best_score:
                    best_x = candidate
                    best_score = score
    
    # Final verification: ensure positive and sorted
    best_x = np.sort(best_x)
    best_x = np.maximum(best_x, 0.1)  # Ensure strictly positive
    
    return best_x.tolist()
# EVOLVE-BLOCK-END
