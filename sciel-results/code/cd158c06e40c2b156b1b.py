import numpy as np
from scipy.optimize import minimize

N = 100

def _correlations(h):
    # Compute all correlations
    # np.correlate(h, 1-h, mode='full')
    return np.correlate(h, 1 - h, mode='full')

def _objective(params):
    h = params
    c = _correlations(h)
    # Soft max to smooth the max
    # Avoid overflow by subtracting max
    c_max = np.max(c)
    if c_max > 0:
        log_sum_exp = c_max + np.log(np.sum(np.exp((c - c_max) / 1.0)))
    else:
        log_sum_exp = np.log(np.sum(np.exp(c)))
    return log_sum_exp * 1.0

def construct():
    # Initial seed
    h0 = np.full(N, 0.5)
    
    # Constraints
    # Sum h = N/2
    constraints = [
        {
            'type': 'eq',
            'fun': lambda x: np.sum(x) - N / 2.0,
            'jac': lambda x: np.ones(N)
        }
    ]
    
    # Bounds
    bounds = [(0.0, 1.0)] * N
    
    # Optimize
    # Use L-BFGS-B for speed, but it doesn't handle equality constraints well directly in older versions?
    # SLSQP handles constraints.
    # Given N=100, SLSQP should be fine within 120s.
    
    res = minimize(
        _objective, 
        h0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-8}
    )
    
    h_opt = res.x
    
    # Ensure constraints are met strictly
    # Project to sum N/2
    diff = N / 2.0 - np.sum(h_opt)
    h_opt += diff / N
    
    # Clip to [0, 1]
    h_opt = np.clip(h_opt, 0.0, 1.0)
    
    # Re-project to sum N/2 after clipping?
    # Simple iterative projection
    for _ in range(10):
        h_opt = np.clip(h_opt, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h_opt)
        h_opt += diff / N
    
    return h_opt
