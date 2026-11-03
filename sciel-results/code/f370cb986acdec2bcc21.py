"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _metric(x):
    """Compute max_k |c_k| / (sum x)^2 where c is linear autoconvolution."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    # Linear autoconvolution via FFT
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    C = np.fft.irfft(X * X, n=n_fft)
    max_abs = np.max(np.abs(C))
    return max_abs / (s * s)

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # 10s safety margin before 120s limit
    
    best_x = np.ones(N)
    best_val = _metric(best_x)  # baseline: all ones -> max_conv = N, sum^2 = N^2, val = 1/N
    
    # Strategy: signed binary sequences (±1) with sum close to 0 or optimized
    # The metric is max|c_k| / (sum x)^2
    # For ±1 sequences, sum x is even, and we want sum large in magnitude relative to peak conv
    # Actually, if sum is large, denominator is large -> good
    # But if all same sign, conv peak is N (at lag 0) -> val = N/N^2 = 1/N
    # If we mix signs, we can reduce peak conv while keeping sum reasonable
    
    # Let's try: sequences with a block of +1s and a block of -1s
    # Or: random ±1 with optimization
    
    # Phase 1: Try structured patterns
    # Pattern: k positive then (N-k) negative
    for k in range(1, N):
        x = np.ones(N)
        x[k:] = -1
        val = _metric(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        if time.time() > deadline:
            return best_x
    
    # Pattern: alternating blocks
    for block_len in range(2, N+1, 2):
        x = np.ones(N)
        for i in range(0, N, block_len):
            end = min(i + block_len//2, N)
            x[i:end] = 1
            x[end:min(i+block_len, N)] = -1
        val = _metric(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        if time.time() > deadline:
            return best_x
    
    # Phase 2: Local search on binary ±1 sequences
    # Start from best found so far, or random
    x = best_x.copy()
    current_val = best_val
    
    # Greedy local search: flip individual bits if it improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            old_val = x[i]
            x[i] = -old_val
            new_val = _metric(x)
            if new_val < current_val:
                current_val = new_val
                best_val = new_val
                best_x = x.copy()
                improved = True
            else:
                x[i] = old_val
        if time.time() > deadline:
            break
    
    # Phase 3: Try random restarts with local search
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 50:
        x = np.ones(N)
        # Random subset to flip to -1
        n_neg = random.randint(N//4, 3*N//4)
        indices = random.sample(range(N), n_neg)
        x[indices] = -1
        
        # Quick local search
        cur = _metric(x)
        for _ in range(200):
            if time.time() > deadline:
                break
            i = random.randint(0, N-1)
            old_val = x[i]
            x[i] = -old_val
            new_val = _metric(x)
            if new_val < cur:
                cur = new_val
            else:
                x[i] = old_val
        
        if cur < best_val:
            best_val = cur
            best_x = x.copy()
        attempts += 1
    
    return best_x
# EVOLVE-BLOCK-END
