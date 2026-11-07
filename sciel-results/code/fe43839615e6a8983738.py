import numpy as np
import time
from scipy.optimize import minimize

def construct():
    N = 100
    deadline = time.time() + 140  # 10s safety margin
    
    def objective(params):
        h = np.clip(params, 0, 1)
        # Enforce sum constraint by adjusting
        current_sum = np.sum(h)
        if current_sum > 0:
            h = h * (N / 2) / current_sum
            h = np.clip(h, 0, 1)
        corr = np.correlate(h, 1 - h, mode="full")
        # Use soft-max approximation for smoothness
        alpha = 10.0
        return np.log(np.sum(np.exp(alpha * corr))) / alpha
    
    # Initial guess: sinusoidal
    i = np.arange(N)
    h0 = 0.5 + 0.5 * np.cos(2 * np.pi * i / N)
    
    best_h = h0.copy()
    best_val = np.max(np.correlate(best_h, 1 - best_h, mode="full")) / N * 2
    
    # Try optimization
    try:
        result = minimize(objective, h0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-12})
        if result.success and time.time() < deadline:
            h_opt = np.clip(result.x, 0, 1)
            # Enforce sum constraint
            h_opt = h_opt * (N / 2) / np.sum(h_opt)
            h_opt = np.clip(h_opt, 0, 1)
            val = np.max(np.correlate(h_opt, 1 - h_opt, mode="full")) / N * 2
            if val < best_val:
                best_h = h_opt
                best_val = val
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 5:
        return best_h
    
    # Try a few more random restarts if time permits
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        h_rand = np.random.rand(N)
        h_rand = h_rand * (N / 2) / np.sum(h_rand)
        h_rand = np.clip(h_rand, 0, 1)
        try:
            result = minimize(objective, h_rand, method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12})
            if result.success:
                h_opt = np.clip(result.x, 0, 1)
                h_opt = h_opt * (N / 2) / np.sum(h_opt)
                h_opt = np.clip(h_opt, 0, 1)
                val = np.max(np.correlate(h_opt, 1 - h_opt, mode="full")) / N * 2
                if val < best_val:
                    best_h = h_opt
                    best_val = val
        except Exception:
            pass
    
    return best_h
