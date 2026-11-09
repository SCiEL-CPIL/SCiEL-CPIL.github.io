import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 100

def calculate_max_overlap(h):
    # h is numpy array of length N
    # Calculate linear max overlap
    # O_k = sum_{i=k}^{N-1} h[i] * (1 - h[i-k])
    # We want max over k in 0..N-1
    # Note: k=0 is sum h[i](1-h[i])
    
    max_ov = 0
    # Vectorized calculation for all k is expensive (N^2)
    # But N=100, so 100*100 = 10000 ops, very fast.
    
    # We can use np.correlate or explicit loops
    # For k=0:
    ov = np.sum(h * (1 - h))
    if ov > max_ov: max_ov = ov
    
    for k in range(1, N):
        # h[i] for i=k..N-1
        # h[i-k] for i=k..N-1 -> h[0..N-1-k]
        prod = h[k:] * (1 - h[:-k])
        ov = np.sum(prod)
        if ov > max_ov:
            max_ov = ov
            
    return max_ov

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Base solution
    h_base = np.full(N, 0.5)
    base_score = calculate_max_overlap(h_base)
    
    best_h = h_base
    best_score = base_score
    
    # Define function to evaluate a candidate defined by sigma and amp
    # We will search for sigma in [5, 50] and amp in [0, 0.49]
    
    def objective(params):
        sigma, amp = params
        x = np.arange(N)
        center = N / 2.0
        # Gaussian peak
        eps = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        eps -= np.mean(eps)
        eps_max = np.max(np.abs(eps))
        if eps_max == 0:
            return 100.0
        eps = eps / eps_max
        
        h_candidate = 0.5 + amp * eps
        
        # Check bounds
        if np.min(h_candidate) < 0 or np.max(h_candidate) > 1:
            return 100.0
            
        # Correct sum
        diff = N / 2.0 - np.sum(h_candidate)
        # Adjust the element closest to 0.5 to maintain bounds
        # Or distribute? Distributing might break smoothness.
        # Single point adjustment is safer for bounds.
        idx = np.argmin(np.abs(h_candidate - 0.5))
        h_candidate[idx] += diff
        
        if np.min(h_candidate) < 0 or np.max(h_candidate) > 1:
            return 100.0
            
        return calculate_max_overlap(h_candidate)

    # Grid search for initial good parameters
    # Then local optimization
    
    best_params = None
    min_obj = 100.0
    
    # Coarse grid
    sigmas = np.linspace(5, 40, 8)
    amps = np.linspace(0.0, 0.45, 5)
    
    for s in sigmas:
        for a in amps:
            val = objective([s, a])
            if val < min_obj:
                min_obj = val
                best_params = [s, a]
                
    # Local optimization using scipy
    if best_params:
        try:
            res = minimize_scalar(lambda x: objective([x, best_params[1]]), bounds=(best_params[0]*0.5, best_params[0]*1.5), method='bounded')
            if res.success:
                s_opt = res.x
                # Optimize amp
                res2 = minimize_scalar(lambda x: objective([s_opt, x]), bounds=(0, 0.49), method='bounded')
                if res2.success:
                    a_opt = res2.x
                    final_val = objective([s_opt, a_opt])
                    if final_val < min_obj:
                        min_obj = final_val
                        best_params = [s_opt, a_opt]
        except Exception:
            pass

    # Construct final best h
    if best_params:
        sigma, amp = best_params
        x = np.arange(N)
        center = N / 2.0
        eps = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        eps -= np.mean(eps)
        eps_max = np.max(np.abs(eps))
        if eps_max > 0:
            eps = eps / eps_max
        h_candidate = 0.5 + amp * eps
        diff = N / 2.0 - np.sum(h_candidate)
        idx = np.argmin(np.abs(h_candidate - 0.5))
        h_candidate[idx] += diff
        if np.min(h_candidate) >= 0 and np.max(h_candidate) <= 1:
            best_h = h_candidate
            best_score = min_obj
            
    return best_h
