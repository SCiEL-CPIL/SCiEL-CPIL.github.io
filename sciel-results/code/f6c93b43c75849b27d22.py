import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Choose L to balance time and score
    # L=4e6 -> K~2.5e4 -> Pairs~3e8 -> ~30s
    L = 4_000_000
    
    powerful_set = set()
    max_v = int(L ** (1/3)) + 1
    for v in range(1, max_v + 1):
        v3 = v ** 3
        max_u = isqrt(L // v3)
        for u in range(1, max_u + 1):
            m = (u * u) * v3
            powerful_set.add(m)
            
    powerful = sorted(powerful_set)
    K = len(powerful)
    
    max_sum = 2 * L
    counts = [0] * (max_sum + 1)
    
    P = powerful
    cnt = counts
    # Optimization: pre-calculate P[j] access? No.
    # Use local variables for speed
    for i in range(K):
        a = P[i]
        # b >= a
        # a + b <= max_sum
        # b <= max_sum - a
        # Find index of last b <= max_sum - a
        # Since P is sorted, we can binary search or just loop and break
        # Loop and break is fine since we are already looping
        for j in range(i, K):
            b = P[j]
            s = a + b
            if s > max_sum:
                break
            cnt[s] += 1
            
    best_N = 0
    best_count = 0
    for n in range(2, max_sum + 1):
        c = cnt[n]
        if c > best_count:
            best_count = c
            best_N = n
            
    pairs = []
    p_set = set(powerful)
    for a in powerful:
        if a > best_N:
            break
        b = best_N - a
        if b >= a and b in p_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
