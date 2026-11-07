import numpy as np
import scipy.optimize as opt
import time

N = 100

def _raw_value(h):
    # h is 1D array of length N
    # value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Initial guess: sinusoid
    # Try a few different periods
    best_h = np.full(N, 0.5)
    best_val = _raw_value(best_h)
    
    # Generate some diverse initializations
    inits = []
    inits.append(np.full(N, 0.5))
    for P in [10, 20, 50, 100]:
        t = np.arange(N)
        h_init = 0.5 + 0.5 * np.cos(2 * np.pi * t / P)
        # Clip to [0,1] just in case, though amplitude 0.5 is safe
        h_init = np.clip(h_init, 0, 1)
        # Adjust sum to 50
        diff = 50 - np.sum(h_init)
        h_init += diff / N
        h_init = np.clip(h_init, 0, 1)
        inits.append(h_init)
        
    # Add a random init
    np.random.seed(42)
    h_rand = np.random.rand(N)
    h_rand = (h_rand - np.mean(h_rand)) * 0.5 + 0.5
    h_rand = np.clip(h_rand, 0, 1)
    diff = 50 - np.sum(h_rand)
    h_rand += diff / N
    h_rand = np.clip(h_rand, 0, 1)
    inits.append(h_rand)

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2},
        {'type': 'ineq', 'fun': lambda x: x}, # x >= 0
        {'type': 'ineq', 'fun': lambda x: 1 - x} # x <= 1
    ]
    
    for h0 in inits:
        if time.time() > deadline:
            break
        
        # SLSQP optimization
        try:
            res = opt.minimize(
                _raw_value, 
                h0, 
                method='SLSQP', 
                bounds=[(0, 1)] * N, 
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            if res.success and _is_valid(res.x):
                val = _raw_value(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x
        except Exception:
            pass

    return best_h
