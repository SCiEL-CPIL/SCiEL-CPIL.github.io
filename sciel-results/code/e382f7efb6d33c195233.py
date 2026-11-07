import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 10000000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline:
        return [11, 6]
        
    best_error = float('inf')
    best_pair = (11, 6) # Fallback
    
    # We iterate k in order of magnitude: 0, 1, -1, 2, -2, ...
    # k = 2a - 3b
    # If k=0, a/b = 3/2 exactly, error 0, which is INVALID.
    # So we only update best_pair for k != 0.
    
    # Max possible b for any k is roughly 2N/3
    # Min possible error for a given k is |k| / (2 * b_max)
    
    max_k_to_check = 100 # Hard cap, should be reached naturally
    
    for k_mag in range(0, max_k_to_check + 1):
        for k in ([k_mag, -k_mag] if k_mag > 0 else [0]):
            
            # Theoretical min error for this k
            # b_max is approx min(N, (2N - k) // 3)
            b_max_est = min(N, (2 * N - k) // 3)
            if b_max_est < 1:
                continue
                
            min_possible_error = abs(k) / (2.0 * b_max_est)
            
            # If k is 0, error is 0, which is invalid. 
            # We don't use it to update best_error, but we should still check 
            # if a solution exists? No, because it's invalid. 
            # However, finding a k=0 solution doesn't help us find a better 
            # non-zero k solution directly, except that we know the sigma 
            # values match for those specific a,b.
            # But we are searching for the *best valid* pair.
            # So for k=0, we skip updating best_pair.
            
            # Check if it's worth processing
            # If k != 0 and min_possible_error >= best_error, we can skip/break?
            # For fixed k_mag, as k_mag increases, min_possible_error increases.
            # So if min_possible_error >= best_error for current k, 
            # it will also be >= for all larger k_mag.
            # So we can break the outer loop.
            if k != 0 and min_possible_error >= best_error:
                break
            
            # Determine valid range for b
            # 1 <= (3b + k) / 2 <= N
            # 2 - k <= 3b <= 2N - k
            b_min = max(1, (2 - k + 2) // 3) # Ceiling of (2-k)/3
            # Note: (2-k) might be negative.
            # If 2-k <= 0, b_min = 1 is sufficient as b>=1.
            # Ceiling division for positive divisor: (num + den - 1) // den
            # But 2-k can be negative.
            # Let's just use:
            # 3b >= 2-k => b >= (2-k)/3
            # If 2-k > 0: b_min = (2-k + 2) // 3
            # If 2-k <= 0: b_min = 1
            
            if 2 - k > 0:
                b_min = (2 - k + 2) // 3
            else:
                b_min = 1
                
            b_max = min(N, (2 * N - k) // 3)
            
            if b_min > b_max:
                continue
                
            # Create array of b values
            # Step size 1, then filter by parity
            # To save memory/time, we can generate with correct parity directly?
            # np.arange doesn't support parity step directly easily without mask.
            # Masking is fine.
            
            b_vals = np.arange(b_min, b_max + 1, dtype=np.int64)
            
            # Parity filter: 3b + k must be even => b + k must be even
            if k % 2 == 0:
                # b must be even
                # Find first even >= b_min
                if b_min % 2 != 0:
                    b_min += 1
                if b_min > b_max:
                    continue
                b_vals = np.arange(b_min, b_max + 1, 2, dtype=np.int64)
            else:
                # b must be odd
                if b_min % 2 == 0:
                    b_min += 1
                if b_min > b_max:
                    continue
                b_vals = np.arange(b_min, b_max + 1, 2, dtype=np.int64)
                
            if len(b_vals) == 0:
                continue
                
            a_vals = (3 * b_vals + k) // 2
            
            # Check sigma match
            matches = sigma[a_vals] == sigma[b_vals]
            
            if np.any(matches):
                # We want the largest b to minimize error |k|/2b
                # indices where matches is True
                # Since b_vals is sorted ascending, the last True is the max b
                # We can use np.where or just find the last index
                # For efficiency, we can check from the end?
                # But np.any is fast.
                # To get max b:
                valid_b = b_vals[matches]
                # valid_b is sorted ascending
                best_b = valid_b[-1]
                # Find corresponding a
                # Since b_vals is unique, we can just compute a
                best_a = (3 * best_b + k) // 2
                
                if k != 0:
                    error = abs(k) / (2.0 * best_b)
                    if error < best_error:
                        best_error = error
                        best_pair = (int(best_a), int(best_b))
                        
        if time.time() > deadline:
            break

    a, b = best_pair
    # Final validation
    if a == b or sigma[a] != sigma[b]:
        return [11, 6]
        
    return [a, b]
