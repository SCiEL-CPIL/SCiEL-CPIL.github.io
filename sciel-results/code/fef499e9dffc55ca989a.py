"""
Maximum sum of pairwise-coprime integers in [1, 256].

Key insight:
- For primes p > 16, p^2 > 256, so the only power of p in range is p itself.
- These large primes never conflict with each other or with any product of small primes.
- So all large primes are always included.
- For small primes {2,3,5,7,11,13}, we partition them into subsets whose products are <= 256,
  maximizing the sum of products.
- Additionally, we can include 1 (coprime to everything, doesn't consume any prime).

This reduces to a small combinatorial optimization over 6 elements, solvable by DP.
"""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 256
    
    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Small primes: those where p^2 <= 256 or p <= 16
    # Actually, for p > 16, p^2 > 256, so only p itself is a candidate
    # Small primes are 2, 3, 5, 7, 11, 13
    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 13]
    
    # All large primes are always included
    base_selection = list(large_primes)
    base_sum = sum(large_primes)
    
    # DP over partitions of small primes
    n = len(small_primes)
    full_mask = (1 << n) - 1
    
    # Precompute product for each mask
    mask_product = [0] * (1 << n)
    mask_valid = [False] * (1 << n)
    mask_product[0] = 0
    mask_valid[0] = True
    
    for mask in range(1, 1 << n):
        # Find lowest set bit
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        if mask_valid[prev]:
            prod = mask_product[prev] * small_primes[idx]
            if prod <= N:
                mask_product[mask] = prod
                mask_valid[mask] = True
    
    # dp[mask] = max sum of products for a partition of primes in mask
    dp = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        best = 0
        sub = mask
        while sub:
            if mask_valid[sub] and dp[mask ^ sub] > 0:
                val = mask_product[sub] + dp[mask ^ sub]
                if val > best:
                    best = val
            sub = (sub - 1) & mask
        dp[mask] = best
    
    max_small_sum = dp[full_mask]
    
    # Reconstruct the partition
    def reconstruct(mask, dp, mask_valid, mask_product):
        if mask == 0:
            return []
        best_val = 0
        best_sub = 0
        sub = mask
        while sub:
            if mask_valid[sub] and dp[mask ^ sub] > 0:
                val = mask_product[sub] + dp[mask ^ sub]
                if val > best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        
        if best_sub == 0:
            # Single prime case
            return [mask_product[mask]]
        
        result = [mask_product[best_sub]]
        result.extend(reconstruct(mask ^ best_sub, dp, mask_valid, mask_product))
        return result
    
    small_selection = reconstruct(full_mask, dp, mask_valid, mask_product)
    
    # Combine
    final_selection = base_selection + small_selection + [1]
    
    # Ensure distinctness and validity
    # Check for duplicates
    seen = set()
    clean = []
    for x in final_selection:
        if x not in seen and 1 <= x <= N:
            seen.add(x)
            clean.append(x)
    final_selection = clean
    
    # Verify pairwise coprimality
    valid = True
    for i in range(len(final_selection)):
        for j in range(i + 1, len(final_selection)):
            if math.gcd(final_selection[i], final_selection[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: all prime powers
        result = [1]
        for p in primes:
            power = p
            while power * p <= N:
                power *= p
            result.append(power)
        return [int(x) for x in result]
    
    return [int(x) for x in final_selection]

# EVOLVE-BLOCK-END
