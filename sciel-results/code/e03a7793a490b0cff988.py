import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    # N = 10^7 is safe and fast with NumPy
    N = 10**7
    
    # Initialize phi
    phi = np.arange(N + 1, dtype=np.int64)
    
    # Sieve
    for i in range(2, N + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
            
        if time.time() - start_time > DEADLINE - 30:
            break
            
    # Sliding window for distinct totients
    # Max phi value is N-1
    last_occ = np.zeros(N + 1, dtype=np.int32)
    
    best_k = 1
    best_s = 1
    current_start = 1
    
    for i in range(1, N + 1):
        val = int(phi[i])
        
        if last_occ[val] >= current_start:
            current_start = last_occ[val] + 1
            
        last_occ[val] = i
        
        k = i - current_start + 1
        if k > best_k:
            best_k = k
            best_s = current_start
            
        if time.time() - start_time > DEADLINE - 5:
            break
            
    if best_k > 200:
        best_k = 200
        
    return {'start': best_s, 'length': best_k}
