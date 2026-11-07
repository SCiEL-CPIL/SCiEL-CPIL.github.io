import numpy as np
import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 10_000_000
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        
    if time.time() - t0 > DEADLINE:
        return [11, 6]

    best_err = float('inf')
    best_pair = (11, 6)
    
    # We iterate b from N down to 1
    # We check k values
    # k = 2a - 3b
    # a = (3b + k) / 2
    
    for b in range(N, 0, -1):
        s_b = sigma[b]
        
        # Max k allowed by current best error
        if best_err == float('inf'):
            max_k = 2 * N # Arbitrary large, constrained by a <= N
        else:
            max_k = int(2 * b * best_err)
            
        if max_k < 1:
            break
            
        # Also constrain by a <= N
        # (3b + k) / 2 <= N  =>  k <= 2N - 3b
        max_k_a = 2 * N - 3 * b
        if max_k_a < max_k:
            max_k = max_k_a
            
        if max_k < 1:
            # No positive k can produce a valid a <= N
            # We could check negative k here, but let's skip for speed first
            # Or just continue to next b
            continue

        # Determine parity
        if b % 2 == 0:
            start_k = 2
            step_k = 2
        else:
            start_k = 1
            step_k = 2
            
        # Check k from start_k to max_k
        # We can use a while loop
        k = start_k
        found = False
        while k <= max_k:
            a_num = 3 * b + k
            a = a_num // 2 # a_num is guaranteed even
            if 1 <= a <= N:
                if sigma[a] == s_b:
                    err = k / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
                    found = True
                    break # Smallest k is best for this b
            k += step_k
            
        # If we didn't find any for this b, we could check negative k?
        # For now, we assume positive k is sufficient to find good solutions.
        # If time permits, we can add negative k check.
        
        if time.time() - t0 > DEADLINE:
            break

    return [int(best_pair[0]), int(best_pair[1])]
