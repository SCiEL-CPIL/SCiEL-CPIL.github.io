import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0
    N = 10_000_000
    
    # 1. Compute sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    # Vectorized sieve
    for i in range(1, N + 1):
        if time.time() - start_time > deadline - 10:
            # Fallback
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() - start_time > deadline - 5:
        return [11, 6]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Iterate b from N down to 1
    # This prioritizes large b, which allows for smaller errors if numerator is small.
    for b in range(N, 0, -1):
        if time.time() - start_time > deadline:
            break
            
        sb = sigma[b]
        
        # Candidates for a
        # a ~ 1.5 b
        # 2a ~ 3b
        # a = floor(1.5 b) or ceil(1.5 b)
        
        # Let's compute 3b to avoid floating point issues
        three_b = 3 * b
        
        # a1 = floor(1.5 b)
        a1 = three_b // 2
        # a2 = a1 + 1 (if 3b is odd, a1 is floor, a2 is ceil)
        # If 3b is even, a1 is exact integer 1.5b, so a1 is the only candidate?
        # No, if 3b is even, 1.5b is integer. a1 = 1.5b.
        # Then 2*a1 - 3b = 0. This is the exact ratio, which is INVALID.
        # So if 3b is even, a1 is invalid.
        # We must check a1-1 and a1+1?
        # The problem says "Approximate a ratio".
        # If a/b = 3/2 exactly, it is invalid.
        # So if 3b is even, a=1.5b is a candidate but invalid.
        # The next best integers are 1.5b - 1 and 1.5b + 1.
        
        if three_b % 2 == 0:
            # 1.5 b is integer
            # Candidates: a1 - 1, a1 + 1
            candidates = [a1 - 1, a1 + 1]
        else:
            # 1.5 b is x.5
            # Candidates: a1 (floor), a1 + 1 (ceil)
            candidates = [a1, a1 + 1]
            
        for a in candidates:
            if a < 1 or a > N:
                continue
            if a == b:
                continue
            # Check exact ratio
            if 2 * a == three_b:
                continue
            
            if sigma[a] == sb:
                num = abs(2 * a - three_b)
                # den = 2 * b
                # err = num / (2 * b)
                # Since we are iterating b downwards, if num == 1, this is the best possible error for this b.
                # And since b is the largest we've seen with num=1 (if we haven't found one yet),
                # it's a strong candidate.
                
                # Optimization: If num == 1, we can return immediately?
                # If we found num=1 at b_current, error is 1/(2*b_current).
                # Any b' < b_current has min error 1/(2*b') > 1/(2*b_current).
                # So yes, if we find num=1, it is the global optimum.
                if num == 1:
                    return [int(a), int(b)]
                
                # Otherwise, track best
                err = num / (2.0 * b)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
