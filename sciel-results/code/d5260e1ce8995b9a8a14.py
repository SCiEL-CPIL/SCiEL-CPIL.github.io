"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    MAX_N = 256
    
    # 1. Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # 2. Split primes into small (<= 16) and large (> 16)
    # sqrt(256) = 16
    small_primes = [p for p in primes if p <= 16]
    large_primes = [p for p in primes if p > 16]
    
    # 3. Initialize result with 1 and all large primes
    # Large primes are pairwise coprime and coprime to 1.
    selected = [1] + large_primes
    
    # 4. Generate candidates for each small prime
    # A candidate k for prime q must:
    # - be a multiple of q
    # - be <= MAX_N
    # - have NO prime factors > 16 (i.e., be "smooth" w.r.t small_primes)
    # - be coprime to all other selected numbers (checked later)
    
    def is_smooth(k, base_primes):
        temp = k
        for p in base_primes:
            while temp % p == 0:
                temp //= p
        return temp == 1

    candidates = {}
    for q in small_primes:
        cand_list = []
        # Iterate downwards to prioritize larger numbers
        for k in range(MAX_N, 0, -1):
            if k % q == 0 and is_smooth(k, small_primes):
                cand_list.append(k)
        candidates[q] = cand_list

    # 5. Backtracking to select one candidate for each small prime
    # such that they are pairwise coprime, maximizing the sum.
    
    best_sum = 0
    best_selection = {}
    
    # Sort small primes by number of candidates (ascending) to prune faster?
    # Or just fixed order.
    q_order = small_primes
    
    def backtrack(idx, current_sum, current_selection):
        nonlocal best_sum, best_selection
        
        # Time check
        if time.time() > deadline:
            return
        
        if idx == len(q_order):
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = current_selection.copy()
            return
        
        q = q_order[idx]
        # Try each candidate for q
        for k in candidates[q]:
            # Check if k is coprime to all previously selected candidates
            valid = True
            for prev_q, prev_k in current_selection.items():
                if gcd(k, prev_k) != 1:
                    valid = False
                    break
            
            if valid:
                current_selection[q] = k
                backtrack(idx + 1, current_sum + k, current_selection)
                del current_selection[q]
                
                # Optimization: Since candidates are sorted descending, 
                # if we found a valid k, any smaller k will result in lower sum 
                # for this branch IF we don't expect subsequent choices to compensate?
                # No, subsequent choices are independent of the specific value of k 
                # except for coprimality. A smaller k might allow a larger 
                # choice for a later q. So we cannot break early on the first valid k.
                # We must explore all valid k.
                
    backtrack(0, 0, {})
    
    # 6. Add the best selected small-prime multiples to the result
    for q, k in best_selection.items():
        selected.append(k)
        
    # Ensure types are Python int
    result = [int(x) for x in selected]
    
    return result
# EVOLVE-BLOCK-END
