import numpy as np
import scipy.optimize as so
import time

N = 400

def objective(h, beta):
    S = np.sum(h)
    if abs(S) < 1e-10:
        return 1e10
    c = np.convolve(h, h)
    # Soft max
    # c can be large, so scale for numerical stability in exp
    c_max = np.max(c)
    if c_max > 0:
        # log-sum-exp trick
        log_sum_exp = c_max + np.log(np.sum(np.exp((c - c_max) * beta)))
        M_soft = log_sum_exp / beta
    else:
        # If all negative, max is close to 0 or negative
        # But we take max of signed entries.
        # If max is negative, M is negative.
        # V uses abs(2 N M / S^2).
        # If M is negative, abs makes it positive.
        # We want to minimize abs(M).
        # So we want M close to 0.
        # Soft max of negative numbers?
        # If c is all negative, max(c) is the least negative.
        # We want to minimize |max(c)|.
        # So we want max(c) to be close to 0 from below, or small positive.
        # Actually, if max(c) < 0, then |max(c)| = -max(c).
        # Let's just minimize (M_soft)^2 / S^2?
        # Or abs(M_soft) / S^2?
        # The task value is abs(2 N M / S^2).
        # So we minimize abs(M_soft) / S^2.
        
        # If c_max <= 0, exp((c-c_max)*beta) is exp(negative).
        # log_sum_exp will be negative.
        # M_soft will be negative.
        pass
    
    # Standard soft max:
    # To handle negative values and ensure we approximate max(c):
    # If we use standard log-sum-exp, it approximates max(c).
    # We want to minimize abs(2 N * M / S^2).
    
    c_max = np.max(c)
    # Scale c to prevent overflow
    # If c_max is large positive, shift.
    # If c_max is large negative, shift.
    shift = c_max
    vals = np.exp((c - shift) * beta)
    sum_vals = np.sum(vals)
    if sum_vals == 0:
        return 1e10
    log_sum = shift * beta + np.log(sum_vals)
    M_soft = log_sum / beta
    
    val = abs(2 * N * M_soft) / (S * S)
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = objective(best_h, 10.0)
    
    # Try a few initializations
    inits = []
    inits.append(np.ones(N))
    
    # Box of length 200
    h_box = np.zeros(N)
    h_box[:200] = 1.0
    inits.append(h_box)
    
    # Box of length 100
    h_box2 = np.zeros(N)
    h_box2[:100] = 1.0
    inits.append(h_box2)
    
    # Random
    rng = np.random.default_rng(42)
    inits.append(rng.normal(size=N))
    
    # Ramp
    inits.append(np.linspace(1, 0, N))
    
    beta = 10.0
    
    for h0 in inits:
        if time.time() > deadline:
            break
        
        # Normalize h0 to avoid extreme values
        h0 = h0 / np.linalg.norm(h0) * np.sqrt(N) # Keep energy ~ N
        
        res = so.minimize(objective, h0, args=(beta,), method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-12})
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_h = res.x
            
        # Check time
        if time.time() > deadline - 10:
            break
            
    # Final polish with higher beta if time permits
    if time.time() < deadline - 5:
        beta = 100.0
        res = so.minimize(objective, best_h, args=(beta,), method='L-BFGS-B', 
                          options={'maxiter': 50, 'ftol': 1e-12})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_h = res.x
            
    return best_h
