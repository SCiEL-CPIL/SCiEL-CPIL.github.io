"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    MAX_N = 10_000_000
    TARGET_NUM, TARGET_DEN = 3, 2  # target ratio 3/2
    
    # Sieve for sigma(n) for all n up to MAX_N
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i
    
    # Group indices by their sigma value
    # We want pairs (a, b) with sigma[a] == sigma[b] and a/b close to 3/2
    # Best case: a = 3k, b = 2k, so sigma[3k] == sigma[2k]
    
    best_error = None
    best_pair = (11, 6)  # fallback
    
    # First pass: look for exact ratio a/b = 3/2, i.e., a=3k, b=2k
    # This requires sigma[3k] == sigma[2k]
    # 3k <= MAX_N => k <= MAX_N//3
    # 2k <= MAX_N => k <= MAX_N//2
    max_k = MAX_N // 3
    
    for k in range(1, max_k + 1):
        if time.time() > deadline - 5:
            break
        a = 3 * k
        b = 2 * k
        if sigma[a] == sigma[b]:
            # Exact ratio 3/2, error = 0... but that's invalid!
            # The task says raw value of exactly 0 cannot be scored
            # So we need a/b != 3/2 but very close
            pass
    
    # Since exact 0 is invalid, we need a/b close to but not equal to 3/2
    # Strategy: for each b, find a such that sigma[a] == sigma[b] and a/b ≈ 3/2
    # a ≈ 1.5 * b, so a is in range [1.5*b - delta, 1.5*b + delta]
    
    # Better approach: for each sigma value s, collect all n with sigma[n]=s
    # Then for each pair (a,b) in that group, compute |a/b - 3/2|
    # But this could be expensive. Let's be smarter.
    
    # For each b from 1 to MAX_N, the target a is around 3b/2
    # We need sigma[a] == sigma[b]
    # Let's iterate b and check a near 3b/2
    
    # Actually, let's think differently. 
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # So we want to minimize |2a - 3b| subject to sigma[a] == sigma[b]
    
    # For a given difference d = 2a - 3b, we have a = (3b + d)/2
    # So 3b + d must be even, meaning d and 3b have same parity
    # If b is odd, d must be odd; if b is even, d must be even
    
    # Let's try small values of |2a - 3b| = d
    # For d = 1: a = (3b+1)/2, need b odd, a integer
    # For d = 2: a = (3b+2)/2, need b even, a integer  
    # For d = 3: a = (3b+3)/2, need b odd
    # etc.
    
    # Let's search by increasing d = |2a - 3b|
    # For each d, find b such that a = (3b ± d)/2 is integer, 1 <= a,b <= MAX_N
    # and sigma[a] == sigma[b]
    
    # d must satisfy: 3b + d ≡ 0 (mod 2) for a to be integer
    # So d ≡ 3b (mod 2), i.e., d ≡ b (mod 2)
    
    best_abs_diff = None  # |2a - 3b|
    best_pair = (11, 6)
    
    # Search by increasing d
    max_d_to_check = 200  # |2a-3b| up to 200, meaning error up to 200/(2b) which is small for large b
    
    for d in range(1, max_d_to_check + 1):
        if time.time() > deadline - 3:
            break
        
        # Try a = (3b + d) / 2
        # Need 3b + d even => b ≡ d (mod 2)
        # a = (3b + d) // 2, need 1 <= a <= MAX_N
        # b range: a >= 1 => 3b + d >= 2 => b >= (2-d)//3, so b >= 1 for d <= 1
        # a <= MAX_N => 3b + d <= 2*MAX_N => b <= (2*MAX_N - d) // 3
        
        b_min = 1
        b_max = (2 * MAX_N - d) // 3
        
        # b must have same parity as d
        if b_min % 2 != d % 2:
            b_min += 1
        
        # Also need b <= MAX_N
        if b_max > MAX_N:
            b_max = MAX_N
        
        # Try a few b values, focusing on larger b for smaller relative error
        # But we need to check sigma equality
        # Let's sample strategically: try b near MAX_N first for small d
        
        # Actually, let's just iterate through a reasonable range
        # For efficiency, check b in steps or just a subset
        
        found = False
        # Check b from large to small (larger b gives smaller relative error for same d)
        for b in range(b_max, b_min - 1, -1):
            if time.time() > deadline - 2:
                found = True
                break
            if b % 2 != d % 2:
                continue
            a = (3 * b + d) // 2
            if a < 1 or a > MAX_N:
                continue
            if sigma[a] == sigma[b]:
                best_abs_diff = d
                best_pair = (a, b)
                found = True
                break
        
        if found:
            break
        
        # Also try a = (3b - d) / 2
        # Need 3b - d even => b ≡ d (mod 2)
        # a = (3b - d) // 2, need 1 <= a <= MAX_N
        # a >= 1 => 3b - d >= 2 => b >= (d+2)//3
        # a <= MAX_N => 3b - d <= 2*MAX_N => b <= (2*MAX_N + d) // 3
        
        b_min2 = max(1, (d + 2) // 3)
        b_max2 = min(MAX_N, (2 * MAX_N + d) // 3)
        
        if b_min2 % 2 != d % 2:
            b_min2 += 1
        
        if b_min2 <= b_max2:
            for b in range(b_max2, b_min2 - 1, -1):
                if time.time() > deadline - 2:
                    break
                if b % 2 != d % 2:
                    continue
                a = (3 * b - d) // 2
                if a < 1 or a > MAX_N:
                    continue
                if sigma[a] == sigma[b]:
                    best_abs_diff = d
                    best_pair = (a, b)
                    break
            else:
                continue
            break
    
    # If we found a pair with d > 0, the error is d/(2b)
    # Verify our result
    a, b = best_pair
    if sigma[a] != sigma[b]:
        # Fallback
        return [11, 6]
    
    # Compute the actual error to make sure it's not zero
    # |a/b - 3/2| = |2a - 3b| / (2b)
    diff = abs(2 * a - 3 * b)
    if diff == 0:
        # This shouldn't happen since we started d from 1, but just in case
        return [11, 6]
    
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
