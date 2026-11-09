import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 50  # Safety margin for 60s limit
    
    N = 1000000
    max_val = N + 1  # We need lpf for N+1 as well
    
    # Sieve to find Largest Prime Factor using NumPy for speed
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    # For each prime i, mark all multiples with i as their largest prime factor
    # Since we iterate in increasing order, the last assignment gives the LPF
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime - mark all multiples
            lpf[i::i] = i
            
    # Check time
    if time.time() - start_time > deadline:
        return 10000  # Fallback, unlikely to happen
        
    # Convert to list for faster scalar access in the search loop
    lpf_list = lpf.tolist()
    
    min_lpf = float('inf')
    best_n = 10000
    
    # Search for n in [10000, 1000000]
    for n in range(10000, N + 1):
        # Largest prime factor of n(n+1) = max(lpf[n], lpf[n+1])
        current_max = lpf_list[n]
        next_max = lpf_list[n + 1]
        if next_max > current_max:
            current_max = next_max
            
        if current_max < min_lpf:
            min_lpf = current_max
            best_n = n
            
    return best_n
