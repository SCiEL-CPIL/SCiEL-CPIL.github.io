import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # Fast sigma sieve using prime factorization approach
    # sigma[n] = product over p^e || n of (p^(e+1)-1)/(p-1)
    # We can compute this in O(n log log n) using a modified sieve
    
    sigma = [1] * (MAX_N + 1)
    # For each prime p, multiply sigma[m] by (1 + p + p^2 + ... + p^e) for all m divisible by p^e
    # More efficiently: use the multiplicative property
    
    # Actually, let's use the standard O(n log n) but with better constants
    # Or use: sigma[n] = sum_{d|n} d, computed by adding d to all multiples
    
    # The original approach is fine but let's optimize the inner loop
    sigma = [0] * (MAX_N + 1)
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d
    
    best_error = float('inf')
    best_a = None
    best_b = None
    
    # Search for a/b close to 3/2
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # Let d = 2a - 3b, then a = (3b + d)/2, so d must have same parity as 3b
    # For fixed d, we want large b to minimize |d|/(2b)
    
    # Strategy: iterate over small |d| values (1, 2, 3, ...) and for each,
    # find the largest b such that a = (3b+d)/2 is a valid integer in [1, MAX_N]
    # and sigma[a] == sigma[b]
    
    # For d=1: a = (3b+1)/2, so b must be odd. b_max = 2*MAX_N//3 - 1 (adjusted for parity)
    # For d=2: a = (3b+2)/2, so b must be even. b_max = 2*MAX_N//3 - 2
    # etc.
    
    # Generate candidate d values in order of |d|
    # For each d, iterate b from max down to min, check sigma equality
    
    # To limit work, we'll check d in order of |d| and stop when we've spent enough time
    # or when the minimum possible error for remaining d values exceeds current best
    
    # Let's try d values from 1 to some limit, and for each, check b from large to small
    
    # For efficiency, we can precompute a mapping from sigma value to list of numbers
    # Then for each b, we can quickly find all a with sigma[a] = sigma[b]
    
    # But that uses too much memory. Instead, let's just check directly.
    
    # Better approach: for each d, the valid b values form an arithmetic progression
    # d=1: b = 1, 3, 5, 7, ... (odd)
    # d=2: b = 2, 4, 6, 8, ... (even)  
    # d=3: b = 1, 3, 5, 7, ... (odd)
    # d=4: b = 2, 4, 6, 8, ... (even)
    # In general, d and b must have the same parity (since 3b+d must be even)
    
    # Let's iterate d from 1 upward, and for each d, check b values in descending order
    # Stop checking b for a given d when b < some threshold (since error increases as b decreases)
    
    # The error is |d|/(2b), so for fixed d, we want the largest b with sigma[(3b+d)/2] = sigma[b]
    
    # Let's cap the number of b values we check per d to avoid timeout
    # For d=1, max b is about 2*MAX_N/3 ≈ 6.67M. Checking all is too slow.
    # Instead, check b in chunks and break early if time runs out.
    
    max_d = 50  # Check d values up to 50
    b_check_limit = 500000  # Check at most 500k b values per d
    
    for d in range(1, max_d + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # d and b must have same parity
        # a = (3b + d) / 2 must be integer, so 3b + d must be even
        # 3b has same parity as b, so b + d must be even, i.e., b and d have same parity
        
        # Find valid b range
        # 1 <= a <= MAX_N => 1 <= (3b+d)/2 <= MAX_N => 2-d <= 3b <= 2*MAX_N-d
        # b >= ceil((2-d)/3) and b <= floor((2*MAX_N-d)/3)
        # Also b >= 1
        
        b_min = max(1, math.ceil((2 - d) / 3))
        b_max = min(MAX_N, math.floor((2 * MAX_N - d) / 3))
        
        # Adjust b_min to have correct parity
        if (b_min + d) % 2 != 0:
            b_min += 1
        
        # Iterate b from b_max down to b_min, checking sigma equality
        # But this could be millions of iterations. Let's limit it.
        
        # Check b in descending order, stop after b_check_limit or when error is too large
        checked = 0
        for b in range(b_max, b_min - 1, -1):
            if (b + d) % 2 != 0:
                continue
            
            a = (3 * b + d) // 2
            
            if a < 1 or a > MAX_N:
                continue
            
            if sigma[a] == sigma[b]:
                error = abs(d) / (2 * b)
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b
            
            checked += 1
            if checked >= b_check_limit:
                break
            
            # Early termination: if even the best possible error for smaller b is worse
            # Error = |d|/(2b), so minimum error for this d is |d|/(2*b_min)
            # If |d|/(2*b_min) >= best_error, no smaller b can help
            if abs(d) / (2 * b_min) >= best_error:
                break
    
    if best_a is None:
        # Fallback to seed
        return [11, 6]
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
