import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    N = 10000000
    target_ratio = 1.5
    
    # 1. Compute sigma(n) for n <= N
    # Using the sieve method: sigma[n] = sum of divisors
    # sigma[i::i] += i
    # This is O(N log N) but vectorized.
    
    # Initialize sigma array
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve
    # We can optimize by only iterating up to N
    # The loop i from 1 to N is the bottleneck in Python overhead, 
    # but the vectorized operation is fast.
    # To speed up the Python loop, we can skip non-primes? No, sigma is not multiplicative in a simple sieve way without SPF.
    # The `sigma[i::i] += i` works for all i.
    
    # However, 10^7 iterations of a Python loop might take ~5-10 seconds.
    # Let's try to run it.
    
    for i in range(1, N + 1):
        if time.time() > deadline:
            break
        sigma[i::i] += i
        
    # 2. Search for best pair
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # We check a window of offsets around 1.5 * b
    # a = floor(1.5 * b) + k
    # b ranges from 1 to N
    # a must be <= N
    
    # We vectorize the search for each offset k
    b_arr = np.arange(1, N + 1, dtype=np.int64)
    
    # To save memory and time, we can process in chunks or just rely on vectorization speed.
    # b_arr size 10^7 is 80MB. sigma is 80MB. Total 160MB. OK.
    
    for k in range(-10, 11):
        if time.time() > deadline:
            break
            
        # Calculate candidate a
        # a = 1.5 * b + k
        # 1.5 * b = 3 * b / 2
        # We want integer a.
        # Let's use exact rational arithmetic to avoid float issues in selection, 
        # but for array generation, we can use integer division.
        # a = (3 * b) // 2 + k is not quite right if we want to center.
        # Let's just generate a = (3 * b + 2*k) // 2 ? No.
        # We want to check integers a such that |a/b - 1.5| is small.
        # This is equivalent to |2a - 3b| being small.
        # Let 2a - 3b = d.
        # We want small |d|.
        # a = (3b + d) / 2.
        # So for a fixed d, a is determined by b if (3b+d) is even.
        
        # Let's iterate d from -20 to 20.
        # If d is even, a = (3b+d)/2 is integer.
        # If d is odd, a is not integer for any b?
        # 2a - 3b = d.
        # If d is odd, 2a is even, 3b parity depends on b.
        # If b is even, 3b is even, d must be even.
        # If b is odd, 3b is odd, d must be odd.
        # So for each d, we only consider b with specific parity.
        
        pass # This logic is better.

    # Revised Search Strategy:
    # Iterate d (difference 2a - 3b) from -20 to 20.
    # For each d, find b such that a = (3b + d) / 2 is an integer and 1 <= a <= N.
    # This implies (3b + d) must be even.
    # And 1 <= (3b + d) / 2 <= N  =>  2 - d <= 3b <= 2N - d.
    
    for d in range(-20, 21):
        if time.time() > deadline:
            break
            
        # Determine valid b range
        # 3b >= 2 - d  => b >= ceil((2-d)/3)
        # 3b <= 2N - d => b <= floor((2N-d)/3)
        
        b_min = max(1, (2 - d + 2) // 3) # ceil((2-d)/3)
        b_max = min(N, (2 * N - d) // 3)
        
        if b_min > b_max:
            continue
            
        # Parity condition: (3b + d) % 2 == 0
        # 3b % 2 == b % 2.
        # So (b + d) % 2 == 0 => b % 2 == d % 2.
        
        # Create array of valid b
        # Step size is 2 to maintain parity
        b_start = b_min
        if b_start % 2 != d % 2:
            b_start += 1
            
        if b_start > b_max:
            continue
            
        b_arr = np.arange(b_start, b_max + 1, 2, dtype=np.int64)
        
        # Calculate a
        a_arr = (3 * b_arr + d) // 2
        
        # Filter a <= N and a >= 1 (should be guaranteed by b range)
        valid_mask = (a_arr <= N) & (a_arr >= 1)
        
        if not np.any(valid_mask):
            continue
            
        b_valid = b_arr[valid_mask]
        a_valid = a_arr[valid_mask]
        
        # Get sigma values
        sigma_a = sigma[a_valid]
        sigma_b = sigma[b_valid]
        
        # Find matches
        match_indices = np.where(sigma_a == sigma_b)[0]
        
        if len(match_indices) > 0:
            # Calculate error for these matches
            # error = |a/b - 1.5| = |2a - 3b| / (2b) = |d| / (2b)
            # Since d is fixed, we want to maximize b to minimize error.
            # So we just need the largest b in the match set.
            max_b_idx = np.argmax(b_valid[match_indices])
            b_best = b_valid[match_indices][max_b_idx]
            a_best = a_valid[match_indices][max_b_idx]
            
            # Check if a/b is exactly 1.5 (d=0)
            if d == 0:
                # Invalid if exactly 1.5? 
                # "raw value of exactly 0 cannot be scored"
                # So we skip d=0 matches?
                # Yes, if d=0, error is 0.
                continue
                
            err = abs(d) / (2.0 * b_best)
            
            if err < best_err:
                best_err = err
                best_a = int(a_best)
                best_b = int(b_best)
                
    # If no valid pair found (e.g. all d=0 or none matched), return fallback
    if best_err == float('inf'):
        # Fallback to known pair [11, 6]
        return [11, 6]
        
    return [best_a, best_b]
