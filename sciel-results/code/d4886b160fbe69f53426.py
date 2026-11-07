import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    # Primes to consider for smooth numbers
    # Using more primes increases the chance of finding collisions
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    max_m = 10**18
    
    solutions = defaultdict(list)
    
    # To handle large recursion depths or just use iterative approach?
    # Recursion depth is max number of primes (25), which is fine.
    
    # We will use a recursive generator that builds k and sigma(k) incrementally.
    # k = product p_i^e_i
    # sigma(k) = product (1 + p_i + ... + p_i^e_i)
    
    # Optimization: 
    # 1. Prune if k * sigma(k) > max_m. Since both factors increase, this is safe.
    # 2. We can also prune if k > sqrt(max_m) ? No, because sigma(k) >= k+1, so k*sigma(k) > k^2.
    #    If k > sqrt(max_m), then k^2 > max_m, so k*sigma(k) > max_m.
    #    So we can prune if k > isqrt(max_m).
    #    isqrt(10^18) = 10^9.
    
    sqrt_max_m = math.isqrt(max_m)
    
    def gen_smooth(idx, current_k, current_sigma):
        if time.time() > deadline:
            return

        if idx == len(primes):
            if current_k > 1:
                m_val = current_k * current_sigma
                # We already pruned, but double check for safety
                if m_val <= max_m:
                    solutions[m_val].append(current_k)
            return

        p = primes[idx]
        
        # Option 1: Do not use prime p (exponent 0)
        gen_smooth(idx + 1, current_k, current_sigma)
        
        # Option 2: Use prime p with exponent e >= 1
        # k_new = current_k * p^e
        # sigma_new = current_sigma * (1 + p + ... + p^e)
        
        k_curr = current_k
        sigma_curr = current_sigma
        sum_powers = 1 # 1 + p + ... + p^e
        p_power = 1    # p^e
        
        # We iterate e from 1 upwards
        # Max e is small because p^e grows fast
        # Also k_curr * p^e must be <= sqrt_max_m
        # And sigma_curr * (1+...+p^e) must be such that k*sigma <= max_m
        
        while True:
            # Next exponent e+1
            # Check if we can multiply k by p
            if k_curr > sqrt_max_m // p:
                break
            
            k_curr *= p
            p_power *= p
            sum_powers += p_power
            
            # Check if sigma overflows or if M overflows
            # sigma_curr * sum_powers
            if sum_powers > 0 and sigma_curr > max_m // sum_powers:
                break
                
            sigma_curr_new = sigma_curr * sum_powers
            k_curr_new = k_curr
            
            # Prune: if k_curr_new * sigma_curr_new > max_m, we can break?
            # Yes, because adding more primes or increasing exponent further will only increase M.
            if k_curr_new * sigma_curr_new > max_m:
                break
                
            # Recurse
            gen_smooth(idx + 1, k_curr_new, sigma_curr_new)

    # Start generation
    # Initial k=1, sigma=1
    gen_smooth(0, 1, 1)
    
    # Find best M
    best_m = 0
    best_k_list = []
    
    # Iterate over all found M values
    for m, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_m = m
            best_k_list = k_list
            
    # If we didn't find anything (shouldn't happen), fallback
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Verify and format
    # Ensure distinct
    distinct_k = list(set(best_k_list))
    # Ensure k <= sqrt(M)
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in distinct_k if k <= sqrt_m]
    
    # Sort for determinism
    valid_k.sort()
    
    # Final sanity check: ensure all k in valid_k satisfy k*sigma(k) == best_m
    # This is expensive if we recompute sigma, but we trust the generation.
    # However, the prompt says "checked with exact factorization".
    # My generation is exact.
    
    # One more check: ensure types are correct
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
