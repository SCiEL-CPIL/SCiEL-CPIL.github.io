import numpy as np

def construct():
    # We need to find n in [10000, 1000000] minimizing max(lpf(n), lpf(n+1))
    # lpf(x) is the largest prime factor of x.
    
    N = 1000001
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(N, dtype=np.int64)
    
    # Sieve to compute largest prime factor
    # Initialize with 0.
    # For each prime p, mark all multiples of p with p.
    # Since we iterate p in increasing order, the last prime to mark a number
    # is its largest prime factor.
    
    # We can do this efficiently.
    # lpf[i] = 0 initially.
    for i in range(2, N):
        if lpf[i] == 0: # i is prime
            for j in range(i, N, i):
                lpf[j] = i
                
    # Now lpf[x] contains the largest prime factor of x for all x in [2, N-1]
    # lpf[1] is 0, but we don't care about 1 since n >= 10000.
    
    # We want to minimize max(lpf[n], lpf[n+1]) for n in [10000, 1000000]
    # Note: lpf array index goes up to 1000000.
    # We need lpf[n+1] for n up to 1000000, so we need lpf up to 1000001.
    # My N was 1000001, so indices 0..1000000 are valid.
    # Wait, if n=1000000, we need lpf[1000000] and lpf[1000001].
    # My N is 1000001, so max index is 1000000.
    # I need to extend N to 1000002 to safely access lpf[1000001].
    
    # Let's redo with N = 1000002
    N = 1000002
    lpf = np.zeros(N, dtype=np.int64)
    for i in range(2, N):
        if lpf[i] == 0:
            for j in range(i, N, i):
                lpf[j] = i
                
    # Find the best n
    min_val = np.inf
    best_n = 10000
    
    # Vectorized approach for speed
    # We look at n from 10000 to 1000000
    # lpf[n] and lpf[n+1]
    # max_lpf = np.maximum(lpf[10000:1000001], lpf[10001:1000002])
    # idx = np.argmin(max_lpf)
    # best_n = 10000 + idx
    
    # Let's verify indices:
    # n starts at 10000.
    # lpf[n] corresponds to lpf[10000] ... lpf[1000000]
    # lpf[n+1] corresponds to lpf[10001] ... lpf[1000001]
    # Slice 1: 10000 to 1000001 (exclusive) -> length 990001
    # Slice 2: 10001 to 1000002 (exclusive) -> length 990001
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    max_lpf = np.maximum(lpf_n, lpf_np1)
    min_idx = np.argmin(max_lpf)
    best_n = 10000 + min_idx
    
    return best_n
