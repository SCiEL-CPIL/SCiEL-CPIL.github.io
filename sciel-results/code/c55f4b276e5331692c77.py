"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    TIME_LIMIT = 85  # safety margin before 90s kill
    
    # Precompute primality for numbers up to 256 (128+128)
    MAX_SUM = 256
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    def is_pairwise_coprime(sums):
        """Check if all distinct values in sums are pairwise coprime."""
        distinct = list(set(sums))
        n = len(distinct)
        for i in range(n):
            for j in range(i+1, n):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True

    # We know the maximum possible score is 9/5 = 1.8.
    # We search for a configuration where all 9 sums are distinct primes.
    # This guarantees pairwise coprimality.
    
    # Strategy:
    # A: 3 distinct odd numbers
    # B: 3 distinct even numbers
    # Sums: odd + even = odd.
    # If all 9 sums are distinct primes, we are done.
    
    # To speed up, we iterate small values first.
    # A in [1..50] odd, B in [2..50] even.
    
    # Optimization: 
    # For a fixed A, we can check B.
    # But simple nested loops are fine for this small range.
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Check seed first
    sums_seed = [a + b for a in best_A for b in best_B]
    if is_pairwise_coprime(sums_seed):
        best_score = len(set(sums_seed))
        
    # Search for 9 distinct prime sums
    # Use small ranges to ensure quick completion
    max_val = 60
    
    odds = [x for x in range(1, max_val, 2)]
    evens = [x for x in range(2, max_val, 2)]
    
    # Iterate A
    for A in itertools.combinations(odds, 3):
        if time.time() - t_start > TIME_LIMIT - 2:
            break
        A_list = list(A)
        # Iterate B
        for B in itertools.combinations(evens, 3):
            if time.time() - t_start > TIME_LIMIT - 2:
                break
            B_list = list(B)
            
            sums = [a + b for a in A_list for b in B_list]
            
            # Check if all 9 are distinct
            if len(set(sums)) != 9:
                continue
                
            # Check if all are prime
            if all(is_prime[s] for s in sums):
                # Found a perfect solution
                return [A_list, B_list]
            
            # Fallback: check if pairwise coprime (in case of 1 or 2 involved, though primes > 2 are safer)
            # But since we require distinct and prime, this is covered.
            # If we didn't find primes, we might still have a valid coprime set.
            # But 9 distinct primes is the target.
            if is_pairwise_coprime(sums):
                score = len(set(sums))
                if score > best_score:
                    best_score = score
                    best_A = A_list
                    best_B = B_list
                if best_score == 9:
                    return [best_A, best_B]
                    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
