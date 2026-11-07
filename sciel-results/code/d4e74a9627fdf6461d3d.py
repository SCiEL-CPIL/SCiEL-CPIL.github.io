import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess: 0.5
    h0 = np.full(N, 0.5)
    
    # Bounds
    bounds = [(0, 1)] * N
    
    # Constraints
    # Sum(h) = 50
    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - N/2})
    
    def objective(h):
        # Calculate correlation
        # 1-h
        v = 1 - h
        # Full correlation
        c = np.correlate(h, v, mode='full')
        
        # Smooth max using log-sum-exp
        # To avoid overflow, subtract max
        c_max = np.max(c)
        # If c_max is very large, beta * c might overflow.
        # But c is bounded. Max C is around 25.
        # beta=100 is safe.
        beta = 100.0
        # log(sum(exp(beta * (c - c_max)))) + beta * c_max
        # We want to minimize the max, so we minimize this surrogate.
        # The surrogate is close to max(c).
        
        # Check time
        if time.time() > deadline:
            return 1e10  # Penalize timeout
        
        # Compute log-sum-exp
        # c - c_max is <= 0
        exp_c = np.exp(beta * (c - c_max))
        log_sum_exp = np.log(np.sum(exp_c))
        return log_sum_exp + beta * c_max
        
    def jac(h):
        # Gradient of the smooth max
        # d/dh log(sum(exp(beta * c_k))) = beta * sum(exp(beta * c_k) * dc_k/dh) / sum(exp(beta * c_k))
        # = beta * E[c_k] where prob is proportional to exp(beta * c_k)
        # dc_k/dh_i involves h and 1-h.
        # This is complex to derive analytically for 'full' correlation.
        # Use numerical gradient or finite differences?
        # SLSQP can use numerical gradient if jac is None.
        # Given N=100, numerical gradient is 200 evaluations per step.
        # Might be slow.
        # Let's try analytic if possible, or just use BFGS/L-BFGS-B with numerical?
        # SLSQP with numerical gradient might be okay for 100 vars.
        return None

    # Use SLSQP
    res = None
    best_h = h0
    best_val = objective(h0)
    
    try:
        # Method SLSQP handles equality constraints well.
        # maxiter needs to be large enough, but we have a time limit.
        res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 500, 'ftol': 1e-9, 'disp': False})
        
        if res.success or res.fun < best_val:
            best_h = res.x
            best_val = res.fun
            
    except Exception:
        pass
    
    # Check time before polishing
    if time.time() < deadline - 5.0:
        # Polish with exact max
        def exact_objective(h):
            v = 1 - h
            c = np.correlate(h, v, mode='full')
            return np.max(c)
            
        try:
            res2 = minimize(exact_objective, best_h, method='SLSQP', bounds=bounds, constraints=cons,
                            options={'maxiter': 100, 'ftol': 1e-12, 'disp': False})
            if res2.success or res2.fun < best_val:
                best_h = res2.x
                best_val = res2.fun
        except Exception:
            pass

    # Ensure constraints are met precisely
    # Clip to [0, 1]
    best_h = np.clip(best_h, 0, 1)
    # Adjust sum to be exactly 50
    diff = N/2 - np.sum(best_h)
    if np.abs(diff) > 1e-9:
        # Distribute diff evenly
        best_h += diff / N
        # Re-clip and re-adjust if needed
        best_h = np.clip(best_h, 0, 1)
        diff = N/2 - np.sum(best_h)
        if np.abs(diff) > 1e-9:
             # If clipping caused issues, try to adjust only interior points
             mask = (best_h > 0) & (best_h < 1)
             if np.sum(mask) > 0:
                 best_h[mask] += diff / np.sum(mask)
                 best_h = np.clip(best_h, 0, 1)

    return best_h
