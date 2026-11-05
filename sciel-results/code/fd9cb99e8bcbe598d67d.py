import numpy as np
import time
import random

N = 400


def compute_value(h):
    """Compute the objective value: abs(2*N*max(autoconv)/sum(h)^2)"""
    s = np.sum(h)
    if s == 0:
        return np.inf
    autoconv = np.convolve(h, h)
    return abs(2 * N * np.max(autoconv) / (s * s))


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Initialize with a random +/-1 sequence with guaranteed non-zero sum
    # Use a pattern that gives good initial properties
    # Start with a pseudo-random binary sequence
    random.seed(42)
    
    # Try a few different initializations and keep the best
    best_h = None
    best_val = np.inf
    
    for init_seed in range(20):
        h = np.array([1.0 if random.Random(init_seed).random() < 0.5 else -1.0 for _ in range(N)])
        # Ensure non-zero sum; if zero, flip one element
        s = np.sum(h)
        if s == 0:
            h[0] = -h[0]
            s = np.sum(h)
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        return np.ones(N)
    
    h = best_h.copy()
    current_val = best_val
    
    # Simulated Annealing
    T = 5.0  # initial temperature
    T_min = 0.01
    cooling = 0.995
    
    # Precompute indices for efficiency
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Random flip
        idx = random.randint(0, N - 1)
        old_val = h[idx]
        h[idx] = -h[idx]
        
        s = np.sum(h)
        if s == 0:
            h[idx] = old_val
            T *= cooling
            continue
        
        new_val = compute_value(h)
        
        # Accept if better or with probability exp(-delta/T)
        if new_val < current_val:
            current_val = new_val
            # Keep the flip
        else:
            delta = new_val - current_val
            if delta < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta / T)
            if not accept:
                h[idx] = old_val
        
        T *= cooling
        
        # Periodically check if we should restart with a different temperature schedule
        # but keep it simple for now
    
    # Final polish: try local search with smaller steps
    # Try flipping pairs of elements
    if time.time() < deadline - 10:
        for _ in range(5000):
            if time.time() > deadline:
                break
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            h[i] *= -1
            h[j] *= -1
            s = np.sum(h)
            if s == 0:
                h[i] *= -1
                h[j] *= -1
                continue
            new_val = compute_value(h)
            if new_val >= current_val:
                h[i] *= -1
                h[j] *= -1
    
    # Final check: make sure sum is not zero
    s = np.sum(h)
    if s == 0:
        return np.ones(N)
    
    return h
