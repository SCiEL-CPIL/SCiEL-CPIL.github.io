import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 85  # Safety margin for 90s limit
    
    N = 1000000
    max_val = N + 1
    
    # Use NumPy for faster sieve
    lpf = np.zeros(max_val + 2, dtype=np.int32)
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            # Assign i to all multiples of i
            # This is vectorized in NumPy
            lpf[i::i] = i
            
        # Periodic time check to avoid overhead
        if i % 50000 == 0:
            if time.time() - start_time > deadline:
                return 10000

    if time.time() - start_time > deadline:
        return 10000

    min_lpf = float('inf')
    best_n = 10000
    
    # Convert to list for faster scalar access in the tight scan loop
    lpf_list = lpf.tolist()
    
    # Scan for the best pair
    for n in range(10000, N + 1):
        m1 = lpf_list[n]
        m2 = lpf_list[n+1]
        cur = m1 if m1 > m2 else m2
        if cur < min_lpf:
            min_lpf = cur
            best_n = n
            
    return best_n
