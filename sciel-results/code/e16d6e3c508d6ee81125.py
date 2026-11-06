import time
import math

def construct():
    deadline = time.time() + 50.0
    
    N_LOW = 10000
    N_HIGH = 1000000
    M = N_HIGH + 1
    
    # 1. Precompute Smallest Prime Factor (SPF) up to M
    spf = list(range(M + 1))
    spf[0] = 0
    spf[1] = 1
    limit = int(math.isqrt(M))
    for i in range(2, limit + 1):
        if spf[i] == i:
            for j in range(i * i, M + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Precompute Largest Prime Factor (LPF) for all numbers
    # lpf[n] = largest prime factor of n
    lpf = [1] * (M + 1)
    for k in range(2, M + 1):
        p = spf[k]
        rest = k // p
        prev = lpf[rest]
        lpf[k] = p if p >= prev else prev

    # 3. Find the global minimum of max(lpf[n], lpf[n+1])
    # Since we have the full array, we can scan efficiently.
    # To be safe against time limits, we scan the whole range.
    # 1M iterations is fast in Python (< 0.5s).
    
    best_n = N_LOW
    best_L = max(lpf[N_LOW], lpf[N_LOW + 1])
    
    # Optimization: Iterate and check
    for n in range(N_LOW + 1, N_HIGH + 1):
        # Compute max(lpf[n], lpf[n+1])
        l1 = lpf[n]
        l2 = lpf[n + 1]
        L = l1 if l1 > l2 else l2
        if L < best_L:
            best_L = L
            best_n = n
        if time.time() > deadline:
            break

    return best_n
