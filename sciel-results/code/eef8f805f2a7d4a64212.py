def construct():
    import time
    t0 = time.time()
    
    N = 123457
    
    # Sieve for largest prime factor
    # Using a list of integers
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_val = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Symmetric search: a <= N-a => a <= N//2
    limit = N // 2
    for a in range(1, limit + 1):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        
        if current_max < best_val:
            best_val = current_max
            best_a = a
            best_b = b
            
        # Early termination if perfect smoothness found (unlikely but safe)
        if best_val == 1:
            break
            
        # Time check
        if time.time() - t0 > 85:
            break
            
    return [best_a, best_b]
