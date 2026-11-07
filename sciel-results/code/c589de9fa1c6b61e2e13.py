import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    # Initial guess: uniform 0.5
    h0 = np.full(N, 0.5)
    x0 = np.concatenate([h0, [25.0]]) # t initialized to ~max c_k
    
    # Objective: minimize t
    def objective(x):
        return x[-1]
        
    # Constraints
    # 1. Sum(h) = 50
    def sum_constraint(x):
        return np.sum(x[:N]) - 50
        
    # 2. t >= c_k for all k
    # c_k = 50 - A_k
    # A_k = sum_{i} h_i h_{i+k}
    # We define inequality constraints: c_k - t <= 0
    
    def corr_constraints(x):
        h = x[:N]
        t = x[-1]
        # Compute A_k for k=0..N-1
        # np.correlate(h, h, mode='full')
        # A_k is at index N-1+k
        corr = np.correlate(h, h, mode='full')
        A = corr[N-1 : 2*N-1] # Length N
        c = 50 - A
        # Constraints: c_k - t <= 0
        return c - t

    # Bounds
    # h_i in [0, 1]
    # t in [0, 100]
    bounds = [(0, 1)] * N + [(0, 100)]
    
    # Constraints list
    cons = [
        {'type': 'eq', 'fun': sum_constraint},
        {'type': 'ineq', 'fun': lambda x: -corr_constraints(x)} # SLSQP expects fun >= 0 for ineq? 
        # Wait, SLSQP docs: 
        # type 'ineq': fun(x) >= 0
        # So we need t - c_k >= 0
        # My corr_constraints returns c_k - t.
        # So -corr_constraints(x) returns t - c_k.
        # This is correct.
    ]
    
    start_time = time.time()
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=cons, 
            options={'maxiter': 200, 'ftol': 1e-9}
        )
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > 100:
            # Timeout guard, return best found so far? 
            # SLSQP doesn't easily yield intermediate results.
            # We'll just return the result if it's valid.
            pass

        h_opt = res.x[:N]
        
        # Verify validity
        if not np.all(np.isfinite(h_opt)):
            return h0
            
        # Check sum
        if abs(np.sum(h_opt) - 50) > 1e-6:
            # Normalize?
            h_opt = h_opt * (50 / np.sum(h_opt))
            # Clip
            h_opt = np.clip(h_opt, 0, 1)
            # Re-normalize to be safe?
            # If clipping changes sum, we might be invalid.
            # For robustness, if invalid, return seed.
            if abs(np.sum(h_opt) - 50) > 1e-6:
                return h0
                
        return h_opt

    except Exception:
        return h0

# EVOLVE-BLOCK-END
