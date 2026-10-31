import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 145.0 # 150s limit, 5s margin

def compute_ratio(h):
    h = np.array(h)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    # Linear convolution
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(h_conv))
    
    if max_conv < 1e-12:
        return -1.0
        
    return (l2_sq / (l1 * max_conv))

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    h_uni = np.ones(N) / N
    candidates.append(h_uni)
    
    # 2. Spike
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    candidates.append(h_spike)
    
    # 3. Gaussian-like
    x = np.linspace(-1, 1, N)
    h_gauss = np.exp(-5 * x**2)
    h_gauss /= np.sum(h_gauss)
    candidates.append(h_gauss)
    
    # 4. Cosine bump
    h_cos = 0.5 * (1 + np.cos(np.pi * x))
    h_cos /= np.sum(h_cos)
    candidates.append(h_cos)
    
    best_h = candidates[0]
    best_val = compute_ratio(best_h)
    
    for h in candidates[1:]:
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h
            
    # Optimization
    # Use log-transform or bounds to handle non-negativity
    # Minimize -log(ratio)
    def neg_log_ratio(params):
        h = np.exp(params) # Ensures positivity
        # Scale invariance: fix sum to 1 to reduce dimensionality?
        # Ratio is scale invariant, so we can fix l1=1.
        h_norm = h / np.sum(h)
        return -compute_ratio(h_norm)
        
    # Random restarts
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > DEADLINE:
            break
            
        if i == 0:
            x0 = np.log(best_h + 1e-6)
        else:
            x0 = np.random.randn(N) * 0.5
            
        try:
            res = minimize(neg_log_ratio, x0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-10})
            if res.success:
                h_opt = np.exp(res.x)
                h_opt = h_opt / np.sum(h_opt)
                val_opt = compute_ratio(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass

    # Final check
    if time.time() > DEADLINE - 5:
         # Return best found
         pass

    # Ensure non-negative
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
         best_h = np.ones(N) / N
         
    return list(best_h)
