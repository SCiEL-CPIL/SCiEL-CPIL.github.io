import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _compute_value(h):
    """Compute the exact value for the task."""
    if np.sum(h) == 0:
        return float('inf')
    c = np.convolve(h, h)
    return abs(2 * N * np.max(c) / (np.sum(h) ** 2))

def _soft_max(c, beta):
    """Soft maximum approximation."""
    # Shift to avoid overflow
    c_max = np.max(c)
    if c_max == 0:
        return 0
    # Use log-sum-exp trick
    c_shifted = c - c_max
    log_sum_exp = c_max + np.log(np.sum(np.exp(beta * c_shifted)) / len(c)) / beta
    return log_sum_exp

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Objective function for optimization
    # We optimize h such that sum(h) = 1 to simplify the value formula:
    # value = 2 * N * max(conv(h, h))
    # But the optimizer might drift sum(h) away from 1.
    # It's better to include the normalization in the objective or use a constraint.
    # Using a constraint is safer for the exact metric.
    
    def objective(h):
        # Ensure finite
        if not np.all(np.isfinite(h)):
            return 1e10
        # Normalize to sum=1 to keep the scale fixed and avoid trivial scaling solutions
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e10
        h_norm = h / s
        
        # Compute linear autoconvolution
        # Using FFT for speed
        h_pad = np.zeros(2 * N)
        h_pad[:N] = h_norm
        c = np.real(np.fft.ifft(np.fft.fft(h_pad) ** 2))
        c = c[:2*N-1] # Linear convolution length
        
        # Soft max
        beta = 20.0
        return _soft_max(c, beta)
        
    # Constraint: sum(h) = 1
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    
    # Bounds: keep values reasonable to avoid numerical issues
    bounds = [(-10.0, 10.0) for _ in range(N)]
    
    # Try a few random initializations and one chirp
    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Real Chirp
    t = np.arange(N)
    chirp = np.cos(2 * np.pi * t**2 / N)
    inits.append(chirp)
    
    # 3. Random
    rng = np.random.default_rng(42)
    inits.append(rng.normal(size=N))
    
    # 4. Another random
    inits.append(rng.normal(size=N))
    
    for h0 in inits:
        if time.time() > deadline:
            break
            
        # Normalize initial guess
        h0_norm = h0 / np.sum(h0)
        
        try:
            res = minimize(objective, h0_norm, method='SLSQP', constraints=constraints, 
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success:
                h_opt = res.x
                # Re-normalize to be safe
                h_opt = h_opt / np.sum(h_opt)
                val = _compute_value(h_opt)
                
                if val < best_val:
                    best_val = val
                    best_h = h_opt
            else:
                # If optimization fails, check the best point found?
                # SLSQP might not report success but still improve.
                h_opt = res.x
                if np.sum(h_opt) != 0:
                    h_opt = h_opt / np.sum(h_opt)
                    val = _compute_value(h_opt)
                    if val < best_val:
                        best_val = val
                        best_h = h_opt
                        
        except Exception:
            continue
            
    return best_h
