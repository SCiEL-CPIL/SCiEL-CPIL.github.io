import numpy as np
from bisect import bisect_left
from collections import defaultdict
import time

def construct():
    N = 10000000
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    # 1. Compute sigma
    sigma = np.ones(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        # Vectorized addition
        sigma[d::d] += d
        
    # 2. Search
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5) # Initial seed from prompt
    
    # Iterate b from N down to 1
    # We look for a such that sigma(a) == sigma(b) and |a/b - 1.5| is small.
    # Let 2a - 3b = k. Then a = (3b + k) / 2.
    # Error is |k| / b.
    # We want to minimize |k| / b.
    
    # We check small k values.
    # For a given b, we check k = 1, 3, 5, ... up to k_max.
    # k_max is determined by best_err: we only care if |k|/b < best_err => |k| < best_err * b.
    
    for b in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        # If even k=1 gives error >= best_err, no smaller b can improve (since b decreases)
        if 1.0 / b >= best_err:
            break
            
        s_b = sigma[b]
        
        # Max k to check
        k_max = int(best_err * b)
        if k_max < 1:
            k_max = 1 # Always check k=1 if possible, though 1/b < best_err is guaranteed by break above?
            # Actually if 1/b < best_err, then k_max >= 1 is not guaranteed if best_err is tiny.
            # But if 1/b < best_err, then k=1 is a candidate for improvement.
            
        # We check k values.
        # Note: 3b + k must be even for a to be integer.
        # If b is even, 3b is even, so k must be even.
        # If b is odd, 3b is odd, so k must be odd.
        
        # Start k from 1 or 2 depending on parity?
        # Actually k = |2a - 3b|.
        # If b is even, 3b even, 2a even, k even.
        # If b is odd, 3b odd, 2a even, k odd.
        
        start_k = 1 if b % 2 == 1 else 2
        step_k = 2
        
        for k in range(start_k, k_max + 1, step_k):
            # Check a = (3b + k) / 2
            num = 3 * b + k
            if num % 2 == 0:
                a = num // 2
                if 1 <= a <= N and a != b:
                    if sigma[a] == s_b:
                        err = k / b
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                            # Update k_max for subsequent k checks in this b?
                            # Yes, we found a better err, so we can stop checking larger k for this b
                            # because larger k will give larger err for same b.
                            break
                            
            # Check a = (3b - k) / 2
            num = 3 * b - k
            if num > 0 and num % 2 == 0:
                a = num // 2
                if 1 <= a <= N and a != b:
                    if sigma[a] == s_b:
                        err = k / b
                        if err < best_err:
                            best_err = err
                            best_a = a
                            best_b = b
                            break

    return [int(best_a), int(best_b)]
