"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 256
    
    # Identify all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Precompute the best choice for each prime
    # For a prime p, we can pick at most one number from its multiples that are powers of p
    # (Any non-power of p contains another prime factor, which would conflict with that prime's choice)
    # Actually, wait. If I pick p^k, it only consumes prime p.
    # If I pick a number like 6 (2*3), it consumes both 2 and 3.
    # The problem is equivalent to: Maximize sum of selected numbers.
    # Each selected number consumes its set of prime factors.
    # Each prime factor can be consumed by at most one number.
    
    # Let's stick to the Branch and Bound strategy.
    # We can separate primes into "Small" (p <= 16) and "Large" (p > 16).
    # For Small primes, the maximum power $p^k \le 256$ is usually the best choice IF we don't need to combine them.
    # But can we combine small primes?
    # E.g. 2*3=6. Value 6. Vs 2 (val 2) + 3 (val 3) = 5. 6 > 5.
    # E.g. 2*5=10. Value 10. Vs 2 (val 2) + 5 (val 5) = 7. 10 > 7.
    # E.g. 2*7=14. Value 14. Vs 2 (val 2) + 7 (val 7) = 9. 14 > 9.
    # E.g. 3*5=15. Value 15. Vs 3 (val 3) + 5 (val 5) = 8. 15 > 8.
    # E.g. 2*3*5=30. Value 30. Vs 2+3+5=10. 30 > 10.
    # Generally, for small primes, combining them into a square-free product yields a higher value than keeping them separate,
    # because $ab > a+b$ for $a,b \ge 2$.
    # However, we are limited by the value 256.
    # The "Small" primes are 2, 3, 5, 7, 11, 13.
    # The "Large" primes are 17, 19, ..., 251.
    
    # Strategy:
    # 1. For Large Primes (p > 16), the only multiples are p, p^2, ...
    #    Since p > 16, p^2 > 256 for p >= 17? 17^2 = 289 > 256.
    #    So for all Large Primes, the only candidate is the prime itself.
    #    We MUST include all Large Primes because they don't conflict with anything else (their multiples are > 256).
    #    Wait, do they conflict with Small Primes?
    #    A large prime p > 16 does not divide any number formed by small primes (since that number would be composed of small primes).
    #    So all Large Primes are independent and should always be included.
    
    large_primes = [p for p in primes if p > 16]
    small_primes = [p for p in primes if p <= 16]
    
    base_sum = sum(large_primes)
    base_selection = list(large_primes)
    
    # 2. For Small Primes, we need to decide how to group them.
    #    The available "slots" are the small primes: 2, 3, 5, 7, 11, 13.
    #    We can form any subset of these primes, multiply them to get a number <= 256, and take that number.
    #    We want to partition the set of small primes into disjoint subsets, where each subset's product is <= 256,
    #    maximizing the sum of the products.
    #    Note: We can also leave a prime unused? No, we want to maximize sum. 
    #    Is it ever better to leave a prime unused?
    #    If we have prime p, and we don't use it, we gain 0. If we use it as p, we gain p > 0.
    #    So every small prime must be in exactly one group.
    #    
    #    So the problem reduces to: Partition {2, 3, 5, 7, 11, 13} into subsets S_1, ..., S_k
    #    such that product(S_i) <= 256 for all i, maximizing sum(product(S_i)).
    #    Also, we can include 1 (value 1). 1 is coprime to everything. It doesn't consume any prime.
    #    So we always add 1.
    
    # Let's solve this small partition problem via exhaustive search (DP or Backtracking).
    # The set is small: {2, 3, 5, 7, 11, 13}. Size 6.
    # Number of subsets of 6 elements is 2^6 = 64.
    # We can use a bitmask DP.
    # dp[mask] = max sum for the primes represented by mask.
    
    small_p_list = [2, 3, 5, 7, 11, 13]
    n_small = len(small_p_list)
    full_mask = (1 << n_small) - 1
    
    # Precompute product and validity for all masks
    mask_val = [0] * (1 << n_small)
    mask_valid = [False] * (1 << n_small)
    mask_val[0] = 0
    mask_valid[0] = True
    
    for mask in range(1, 1 << n_small):
        # Get the lowest set bit
        lsb = mask & -mask
        idx = (lsb.bit_length() - 1)
        p = small_p_list[idx]
        prev_mask = mask ^ lsb
        
        if mask_valid[prev_mask]:
            prod = mask_val[prev_mask] * p
            if prod <= N:
                mask_val[mask] = prod
                mask_valid[mask] = True
            else:
                mask_val[mask] = 0
                mask_valid[mask] = False
        else:
            mask_val[mask] = 0
            mask_valid[mask] = False
            
    # DP
    dp = [-1] * (1 << n_small)
    dp[0] = 0
    
    for mask in range(1, 1 << n_small):
        # Iterate through all submasks of mask to find the best partition
        # This is O(3^n) which is 3^6 = 729. Very fast.
        sub = mask
        best = -1
        while sub:
            if mask_valid[sub]:
                if dp[mask ^ sub] != -1:
                    val = mask_val[sub] + dp[mask ^ sub]
                    if val > best:
                        best = val
            sub = (sub - 1) & mask
            
        dp[mask] = best
        
    max_small_sum = dp[full_mask]
    
    # Reconstruct the best selection for small primes
    # We need to find which subsets were chosen.
    # For this specific small case, we can just re-run the logic or store parent pointers.
    # Since it's so small, let's just find one optimal configuration.
    
    def reconstruct(mask):
        if mask == 0:
            return []
        sub = mask
        best_val = -1
        best_sub = -1
        while sub:
            if mask_valid[sub]:
                if dp[mask ^ sub] != -1:
                    val = mask_val[sub] + dp[mask ^ sub]
                    if val > best_val:
                        best_val = val
                        best_sub = sub
            sub = (sub - 1) & mask
            
        if best_sub == -1:
            return [] # Should not happen
            
        # Get the primes in best_sub
        primes_in_sub = []
        for i in range(n_small):
            if best_sub & (1 << i):
                primes_in_sub.append(small_p_list[i])
        
        return [mask_val[best_sub]] + reconstruct(mask ^ best_sub)
        
    small_selection = reconstruct(full_mask)
    
    # Add 1
    final_selection = base_selection + small_selection + [1]
    
    # Verify distinctness and range
    final_selection = list(set(final_selection))
    # All numbers are <= 256 by construction.
    # All numbers are >= 1.
    
    # Final sanity check for pairwise coprimality
    # This should be true by construction, but let's be safe.
    valid = True
    for i in range(len(final_selection)):
        for j in range(i+1, len(final_selection)):
            if math.gcd(final_selection[i], final_selection[j]) != 1:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: Just return the greedy prime powers if something went wrong
        result = [1]
        for p in primes:
            power = p
            while power * p <= N:
                power *= p
            result.append(power)
        return [int(x) for x in result]
        
    return [int(x) for x in final_selection]

# EVOLVE-BLOCK-END
