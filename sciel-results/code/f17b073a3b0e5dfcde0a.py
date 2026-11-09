import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    N = 10000000
    
    # Sieve for sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Using a loop with numpy slicing is efficient
    # But 10^7 iterations in Python loop might be slow.
    # Let's try to optimize the sieve loop.
    # Alternative: 
    # for i in range(1, N+1):
    #     sigma[i::i] += i
    
    # This is O(N log N) operations in C, but O(N) Python loop overhead.
    # 10^7 Python iterations might take ~5-10 seconds.
    # The inner numpy op is fast.
    
    for i in range(1, N + 1):
        if time.time() > deadline - 20.0:
            # If sieve is taking too long, fallback
            # But sieve is the main cost. If it's too slow, we can't finish.
            # Let's assume it fits.
            pass
        sigma[i::i] += i
        
    if time.time() > deadline - 10.0:
        return [11, 6]

    # Search for best pair
    # Error = |k| / (2b) where k = 2a - 3b
    # We want to minimize this.
    # Iterate k from 1 upwards.
    # For each k, find largest b such that a = (3b+k)//2 is valid and sigma(a)==sigma(b).
    
    best_err_num = None
    best_err_den = None
    best_a = None
    best_b = None
    
    # We can limit k. If k is very large, error is large.
    # If we find a solution with error E, we only need to check k such that k/(2N) < E.
    
    # Start with k=1
    # Note: k=0 is invalid (error 0).
    
    # Estimate max k to check.
    # If no solution found for k=1, check k=2, etc.
    # If we find a solution for k=1 with b=1000, error = 1/2000 = 0.0005.
    # Then we only need to check k < 0.0005 * 2N = 10000.
    
    max_k_check = 100000 # Upper bound for k iteration
    
    for k in range(1, max_k_check + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Check if we can stop early
        if best_err_num is not None:
            # Check if k / (2N) >= best_err_num / best_err_den
            # k * best_err_den >= best_err_num * 2N
            if k * best_err_den >= best_err_num * 2 * N:
                break
        
        # Determine valid range for b
        # a = (3b + k) / 2
        # 1 <= a <= N  =>  2 <= 3b + k <= 2N
        # 2 - k <= 3b <= 2N - k
        # ceil((2-k)/3) <= b <= floor((2N-k)/3)
        
        b_min_val = 2 - k
        b_max_val = 2 * N - k
        
        b_min = (b_min_val + 2) // 3  # ceil division for positive, but b_min_val can be negative
        # Correct ceil for any integer:
        # ceil(x/3) = -((-x)//3)
        b_min = -((-b_min_val) // 3)
        b_max = b_max_val // 3
        
        # Clamp to [1, N]
        if b_min < 1: b_min = 1
        if b_max > N: b_max = N
        
        # Parity check: b % 2 == k % 2
        if b_min % 2 != k % 2:
            b_min += 1
        if b_max % 2 != k % 2:
            b_max -= 1
            
        if b_min > b_max:
            continue
            
        # Iterate b from b_max down to b_min
        # We want the largest b to minimize error for this k
        # Using numpy is not helpful here for the loop itself, just for lookup
        for b in range(b_max, b_min - 1, -2):
            a = (3 * b + k) // 2
            # a is int and in [1, N]
            if sigma[a] == sigma[b]:
                # Found a match
                err_num = k
                err_den = 2 * b
                
                # Compare err_num/err_den with best_err_num/best_err_den
                if best_err_num is None or err_num * best_err_den < best_err_num * err_den:
                    best_err_num = err_num
                    best_err_den = err_den
                    best_a = a
                    best_b = b
                break # Found best b for this k
                
    if best_a is not None:
        return [int(best_a), int(best_b)]
        
    # Fallback
    return [11, 6]
