import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n <= N
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # This is the most time-consuming part, but vectorized in NumPy
    for i in range(1, N + 1):
        if i % 100000 == 0 and time.time() > deadline - 20.0:
            # If we are running out of time, we cannot complete the sieve.
            # We must return a valid result. 
            # If sigma is incomplete, we can't trust it.
            # However, the sieve is usually fast enough. 
            # If we hit this, we might have to break and use a fallback?
            # No, breaking the sieve leaves sigma incomplete.
            # We assume the sieve finishes in time.
            pass
        sigma[i::i] += i
        
    # 2. Search for best pair
    # b must satisfy 1.5 * b <= N, so b <= 2N/3
    max_b = int(2 * N / 3)
    
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # Iterate b from max_b down to 1
    # Large b gives smaller potential error (1/2b)
    for b in range(max_b, 0, -1):
        if time.time() > deadline - 5.0:
            break
            
        s_b = sigma[b]
        target = 1.5 * b
        
        # Determine search window for a
        # We want |a/b - 1.5| < best_err
        # |a - 1.5b| < best_err * b
        if best_err == float('inf'):
            # Initial search: check a small window around target
            # to find any valid pair to set a baseline
            window = 10
        else:
            window = int(best_err * b) + 1
            # Cap window to avoid excessive checking if best_err is still large
            if window > 1000:
                window = 1000
        
        a_low = int(target - window)
        a_high = int(target + window)
        
        # Ensure a is within [1, N]
        a_low = max(1, a_low)
        a_high = min(N, a_high)
        
        # Check candidates in the window
        # We prioritize a closer to target
        # Generate indices in order of distance from target
        # For simplicity, just iterate the range. The window is small.
        for a in range(a_low, a_high + 1):
            if sigma[a] == s_b:
                # Calculate error
                # err = abs(a/b - 1.5)
                # Using exact rational comparison is better for precision, 
                # but float is fine for comparison if we are careful.
                # err = abs(2*a - 3*b) / (2*b)
                numerator = abs(2 * a - 3 * b)
                if numerator == 0:
                    continue # Invalid per task: raw value 0 is invalid
                
                # Compare fractions to avoid float precision issues in comparison
                # We want to minimize numerator / (2*b)
                # Current best is best_num / (2*best_b)
                # numerator / best_b_denom < best_num / best_b_denom?
                # Cross multiply: numerator * best_b < best_num * b
                
                # Let's store best_num and best_b for exact comparison
                # Actually, let's just compute float error for simplicity, 
                # as the values are not huge.
                err = numerator / (2.0 * b)
                
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    # We can break out of the inner loop?
                    # No, there might be another a in the window with smaller error?
                    # No, for a fixed b, error is minimized when |2a-3b| is minimized.
                    # The a's in the window are close to 1.5b.
                    # The one with smallest |2a-3b| is the best for this b.
                    # We can break if we find the closest a?
                    # Actually, we should find the a with min |2a-3b| among those with matching sigma.
                    # Since the window is small, we can just check all and keep the best for this b,
                    # then compare with global best.
                    pass
                    
    return [int(best_a), int(best_b)]
