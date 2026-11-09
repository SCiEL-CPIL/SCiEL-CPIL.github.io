import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Generate smallest prime factor (SPF) sieve up to 1000001
    MAX_N = 1000001
    spf = np.zeros(MAX_N + 1, dtype=np.int32)
    
    # Initialize spf for odd numbers (even numbers have spf=2)
    for i in range(2, MAX_N + 1):
        spf[i] = i
    
    spf[0] = 0
    spf[1] = 1
    
    # Sieve: for each prime, mark its multiples
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute largest prime factor for all numbers in [10000, 1000001]
    # Use a faster approach: factor each number using spf
    lpf_arr = np.zeros(MAX_N + 1, dtype=np.int32)
    
    for i in range(10000, MAX_N + 1):
        n = i
        lpf = 1
        while n > 1:
            p = spf[n]
            lpf = p
            while n % p == 0:
                n //= p
        lpf_arr[i] = lpf
    
    # Scan for minimum of max(lpf[n], lpf[n+1])
    # Vectorize the scan
    n_range = np.arange(10000, 1000001)
    max_lpf = np.maximum(lpf_arr[10000:1000001], lpf_arr[10001:1000002])
    
    best_idx = np.argmin(max_lpf)
    best_n = 10000 + best_idx
    best_val = max_lpf[best_idx]
    
    return int(best_n)
