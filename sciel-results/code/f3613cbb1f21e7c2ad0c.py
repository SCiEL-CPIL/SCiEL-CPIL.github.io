import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 120

def construct():
    start_time = time.time()
    
    # Initial candidate: Uniform
    best_h = np.ones(N) / N
    
    def calc_metric(h):
        # Normalize h to sum to 1
        h_sum = np.sum(h)
        if h_sum == 0:
            return float('inf')
        h_norm = h / h_sum
        
        # Use FFT for speed
        f_len = 2048
        f = np.fft.rfft(h_norm, n=f_len)
        c = np.fft.irfft(f * np.conjugate(f), n=f_len)
        return np.max(c)

    best_val = calc_metric(best_h)
    
    # Gradient Descent
    # We optimize h directly, with constraint sum(h)=1 and h>=0
    
    def objective(h):
        # Ensure non-negativity and normalization inside? 
        # Better to use bounds and constraints in optimizer.
        # But for speed, we can just compute metric.
        # The optimizer will handle constraints.
        return calc_metric(h)
    
    # Constraints
    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0})
    
    # Bounds
    bounds = [(0, None) for _ in range(N)]
    
    # Run optimization
    # Limit iterations or time
    # L-BFGS-B is good for bounded problems
    
    try:
        res = minimize(objective, best_h, method='L-BFGS-B', bounds=bounds, constraints=cons, 
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success and res.fun < best_val:
            best_h = res.x
            best_val = res.fun
            
    except Exception:
        pass
        
    # Check time
    if time.time() - start_time < DEADLINE - 5:
        # Try a few random restarts with gradient descent?
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Random start
            h_rand = np.random.dirichlet(np.ones(N))
            try:
                res = minimize(objective, h_rand, method='L-BFGS-B', bounds=bounds, constraints=cons, 
                               options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.success and res.fun < best_val:
                    best_h = res.x
                    best_val = res.fun
            except Exception:
                pass

    # Scale result to sum 600
    if np.sum(best_h) > 0:
        best_h = best_h * (N / np.sum(best_h))
        
    return best_h.tolist()
