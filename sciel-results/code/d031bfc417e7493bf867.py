import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Choose N based on time budget
    # 50 million is a good balance
    N = 50_000_000
    
    # If we run out of time, we can't easily stop the NumPy loop efficiently without checks.
    # But 50M loop in Python is ~10-20s?
    # Actually, `sigma[i::i] += i` is one call.
    # 50 million calls to NumPy is very slow due to overhead.
    # Each call ~1us? 50s.
    # Each call ~10us? 500s.
    # It depends on array size.
    # For i=2, array size 25M.
    # For i=50M, array size 1.
    # Average size is small.
    # But Python loop overhead is constant.
    # 50M iterations is likely > 120s in pure Python loop.
    
    # We need to reduce N or optimize.
    # Let's try N = 10_000_000.
    
    N = 10_000_000
    
    sigma = np.ones(N+1, dtype=np.int64)
    
    # Time check inside loop?
    # If we break, we have partial sigma.
    # Partial sigma is wrong.
    # So we must finish.
    
    # Let's stick to N = 5_000_000 to be safe, or 10_000_000.
    # 10^7 should be under 10s.
    
    for i in range(2, N+1):
        sigma[i::i] += i
        
    k_vals = np.arange(1, N+1, dtype=np.int64)
    M_vals = k_vals * sigma
    
    mask = M_vals <= 10**18
    M_vals = M_vals[mask]
    k_vals = k_vals[mask]
    
    if len(M_vals) == 0:
        return {'M': 1, 'k': [1]}
        
    unique_M, counts, inverse = np.unique(M_vals, return_counts=True, return_inverse=True)
    
    max_idx = np.argmax(counts)
    best_M = unique_M[max_idx]
    k_list = k_vals[inverse == max_idx]
    
    return {'M': int(best_M), 'k': [int(x) for x in k_list]}
