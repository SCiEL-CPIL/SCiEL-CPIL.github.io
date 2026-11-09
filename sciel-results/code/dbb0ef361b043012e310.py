import time
import math
import numpy as np

def construct():
    start_time = time.time()
    MAX_N = 1000000
    MIN_N = 10000
    N_LIMIT = MAX_N + 1

    # 1. Compute Smallest Prime Factor (SPF) using a sieve
    spf = np.arange(N_LIMIT + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if spf[i] == i:
            for j in range(i * i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute powerful part for each number using DP
    p_part = np.ones(N_LIMIT + 1, dtype=np.int64)
    
    for i in range(2, N_LIMIT + 1):
        if spf[i] == i:
            p_part[i] = 1
        else:
            p = spf[i]
            x = i
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                p_part[i] = (p ** e) * p_part[x]
            else:
                p_part[i] = p_part[x]

    # 3. Search for max product
    products = p_part[MIN_N:MAX_N + 1] * p_part[MIN_N + 1:MAX_N + 2]
    max_idx = np.argmax(products)
    best_n = MIN_N + max_idx
    
    # Time guard
    if time.time() - start_time > 50:
        pass # Already computed
        
    return int(best_n)
