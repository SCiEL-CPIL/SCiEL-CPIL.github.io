import time
import numpy as np

def construct():
    deadline = time.time() + 85.0
    
    N = 1000000
    limit = N + 1
    
    # Use numpy for faster sieve? Or stick to list.
    # List is fine.
    
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute LPF for all numbers
    lpf_arr = [1] * (limit + 1)
    for i in range(2, limit + 1):
        x = i
        max_pf = 1
        while x > 1:
            p = spf[x]
            max_pf = p
            while x % p == 0:
                x //= p
        lpf_arr[i] = max_pf
        
    best_n = 10000
    best_val = max(lpf_arr[10000], lpf_arr[10001])
    
    for n in range(10001, N + 1):
        if time.time() > deadline:
            break
        val = max(lpf_arr[n], lpf_arr[n+1])
        if val < best_val:
            best_val = val
            best_n = n
            
    return best_n
