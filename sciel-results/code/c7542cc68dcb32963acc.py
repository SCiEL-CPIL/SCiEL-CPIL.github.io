"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit

    # Primes to use for generating smooth numbers
    # Using more primes to get more combinations
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    n_primes = len(primes)
    
    max_m = 10**18
    
    # Dictionary to store lists of k for each M
    m_to_ks = defaultdict(list)
    
    # Precompute prime powers and sigma contributions for each prime
    # For prime p, we store: p^0, p^1, ..., p^e and corresponding sigma contributions
    prime_powers = []  # prime_powers[i] = list of (p^e, sigma(p^e))
    for p in primes:
        powers = []
        p_pow = 1
        sigma_p = 1
        for e in range(60):
            powers.append((p_pow, sigma_p))
            p_pow *= p
            if p_pow > 10**18:
                break
            sigma_p += p_pow
        prime_powers.append(powers)
    
    # Split primes into two halves for meet-in-the-middle
    mid = n_primes // 2
    left_primes = primes[:mid]
    right_primes = primes[mid:]
    n_left = mid
    n_right = n_primes - mid
    
    # Generate all combinations for left half
    # State: (idx, k, sigma_k)
    left_combos = []
    stack = [(0, 1, 1)]
    count = 0
    
    def time_check():
        return time.time() > deadline
    
    while stack:
        if time_check():
            break
        
        idx, k, sigma_k = stack.pop()
        count += 1
        
        if idx == n_left:
            # We have considered all left primes
            # k * sigma_k must be <= max_m / 1 (since right part contributes at least 1)
            if k * sigma_k <= max_m:
                left_combos.append((k, sigma_k))
            continue
        
        p = left_primes[idx]
        powers = prime_powers[idx]
        
        for p_pow, sigma_p in powers:
            new_k = k * p_pow
            if new_k > 10**18:
                break
            
            new_sigma = sigma_k * sigma_p
            
            # Prune if M exceeds limit
            if new_k * new_sigma > max_m:
                break
            
            stack.append((idx + 1, new_k, new_sigma))
    
    # Sort left combos by k for binary search
    left_combos.sort(key=lambda x: x[0])
    
    # Now for each right combination, find compatible left combinations
    # We need: k_left * k_right * sigma_left * sigma_right <= max_m
    # And: k_total = k_left * k_right, sigma_total = sigma_left * sigma_right
    # M = k_total * sigma_total
    
    # For each right combo, we need k_left * sigma_left <= max_m / (k_right * sigma_right)
    # Also need k_total <= sqrt(M) => k_left * k_right <= sqrt(k_left * k_right * sigma_left * sigma_right)
    # => k_left * k_right <= sqrt(k_left * k_right * sigma_left * sigma_right)
    # => k_left * k_right <= sqrt(k_left * k_right * sigma_left * sigma_right)
    # This simplifies to: k_left * k_right <= sqrt(k_left * k_right * sigma_left * sigma_right)
    # => (k_left * k_right)^2 <= k_left * k_right * sigma_left * sigma_right
    # => k_left * k_right <= sigma_left * sigma_right
    # => k_left / sigma_left <= sigma_right / k_right
    
    # Actually, let's just check the constraint directly
    # k_total <= sqrt(M) means k_total^2 <= M = k_total * sigma_total
    # => k_total <= sigma_total
    # => k_left * k_right <= sigma_left * sigma_right
    
    right_stack = [(0, 1, 1)]
    
    while right_stack:
        if time_check():
            break
        
        idx, k_r, sigma_r = right_stack.pop()
        
        if idx == n_right:
            # For this right combination, find all compatible left combinations
            # M = k_l * k_r * sigma_l * sigma_r <= max_m
            # k_l * k_r <= sigma_l * sigma_r (for k <= sqrt(M))
            # So: k_l * sigma_l <= max_m / (k_r * sigma_r)
            # And: k_l / sigma_l <= sigma_r / k_r
            
            max_left_product = max_m // (k_r * sigma_r)
            if max_left_product < 1:
                continue
            
            # Find left combos where k_l * sigma_l <= max_left_product
            # and k_l * k_r <= sigma_l * sigma_r => k_l / sigma_l <= sigma_r / k_r
            # i.e., k_l * k_r <= sigma_l * sigma_r
            
            # Binary search for upper bound on k_l * sigma_l
            # We need to find all (k_l, sigma_l) in left_combos such that:
            # 1. k_l * sigma_l <= max_left_product
            # 2. k_l * k_r <= sigma_l * sigma_r
            
            # Since left_combos is sorted by k_l, we can iterate through candidates
            # But we need to be efficient. Let's use a different approach.
            
            # For each left combo, check if it's compatible
            # This could be slow, so let's use a smarter approach
            
            # Actually, let's just iterate through left_combos but with early termination
            # Since left_combos is sorted by k_l, and we need k_l * sigma_l <= max_left_product,
            # we can stop when k_l * sigma_l > max_left_product (but sigma_l varies)
            
            # Let's just iterate and check
            for k_l, sigma_l in left_combos:
                if k_l * sigma_l > max_left_product:
                    # Since sorted by k_l, but sigma_l isn't monotonic with k_l
                    # We can't simply break here. But if k_l is too large, 
                    # k_l * sigma_l is likely too large
                    # Actually we can't break safely. Let's continue but this is slow.
                    # For now, let's just check all.
                    pass
                
                # Check constraint: k_l * k_r <= sigma_l * sigma_r
                if k_l * k_r > sigma_l * sigma_r:
                    continue
                
                M = k_l * k_r * sigma_l * sigma_r
                if M <= max_m:
                    k_total = k_l * k_r
                    m_to_ks[M].append(k_total)
            
            continue
        
        p = right_primes[idx]
        powers = prime_powers[n_left + idx]
        
        for p_pow, sigma_p in powers:
            new_k_r = k_r * p_pow
            if new_k_r > 10**18:
                break
            
            new_sigma_r = sigma_r * sigma_p
            
            # Prune if even with minimum left contribution (1*1=1), M exceeds limit
            if new_k_r * new_sigma_r > max_m:
                break
            
            right_stack.append((idx + 1, new_k_r, new_sigma_r))
    
    # Find the best M
    best_m = 0
    best_ks = []
    for M, ks in m_to_ks.items():
        if len(ks) > len(best_ks):
            best_m = M
            best_ks = ks
    
    if not best_ks:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Ensure types are Python int
    best_ks = [int(k) for k in best_ks]
    
    return {'M': int(best_m), 'k': best_ks}
# EVOLVE-BLOCK-END
