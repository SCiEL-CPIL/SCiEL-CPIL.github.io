import numpy as np
from scipy.optimize import minimize

def rudin_problem10():
    # ... setup ...
    best_c = None
    best_val = np.inf
    
    # Try a few initializations
    for _ in range(10):
        c0 = np.random.randn(21)
        c0 /= np.linalg.norm(c0)
        res = minimize(objective, c0, jac=grad, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-10})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_c = res.x
    
    # Also try the "seed" direction? Or Y_10^0?
    # Y_10^0 is index 10 (m=0).
    c0 = np.zeros(21)
    c0[10] = 1.0
    res = minimize(objective, c0, jac=grad, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-10})
    if res.success and res.fun < best_val:
        best_val = res.fun
        best_c = res.x
        
    # Normalize best_c to be safe for validity
    max_c = np.max(np.abs(best_c))
    if max_c == 0:
        best_c = np.ones(21)
    else:
        best_c = best_c / max_c
        
    return best_c
