import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, 5s margin
    
    target = 123457
    max_n = target - 1
    
    # Precompute largest prime factor (LPF) for all numbers up to target-1
    lpf = [0] * (max_n + 1)
    if max_n >= 1:
        lpf[1] = 1
        
    # Sieve to find largest prime factor
    # Complexity O(N log log N)
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                # Since we iterate i in increasing order, the last prime factor
                # assigned will be the largest.
                lpf[j] = i
                
    best_max = float('inf')
    best_a = 1
    best_b = target - 1
    
    # Search for optimal a
    # We only need to search up to target // 2 because the problem is symmetric
    # max(P(a), P(b)) is the same for (a, b) and (b, a)
    limit = target // 2
    for a in range(1, limit + 1):
        # Check time occasionally
        if (a & 0xFF) == 0:
            if time.time() > deadline:
                break
                
        b = target - a
        pa = lpf[a]
        pb = lpf[b]
        cur_max = pa if pa > pb else pb
        
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
            best_b = b
            # Early termination if we reach the theoretical minimum
            # Since target is odd, one of a,b is even (P>=2), one is odd (P>=3 unless 1).
            # If we can get 3, that's the absolute min for odd > 1.
            # But we know the min is likely higher.
            if best_max <= 3:
                break
                
    return [best_a, best_b]
