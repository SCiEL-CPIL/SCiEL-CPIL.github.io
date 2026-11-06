"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    TIME_LIMIT = 85  # safety margin before 90s kill
    
    # Precompute primes up to 256 (max sum 128+128)
    MAX_SUM = 256
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    best_result = [[1, 7, 13], [4, 10, 16]]
    
    # Search for all-prime distinct sumset (cardinality 9)
    # We limit the search to small numbers where primes are denser
    range_limit = 30
    
    for a1 in range(1, range_limit):
        if time.time() - t_start > TIME_LIMIT - 2:
            break
        for a2 in range(a1 + 1, range_limit):
            if time.time() - t_start > TIME_LIMIT - 2:
                break
            for a3 in range(a2 + 1, range_limit):
                if time.time() - t_start > TIME_LIMIT - 2:
                    break
                A = [a1, a2, a3]
                
                for b1 in range(1, range_limit):
                    if time.time() - t_start > TIME_LIMIT - 2:
                        break
                    for b2 in range(b1 + 1, range_limit):
                        if time.time() - t_start > TIME_LIMIT - 2:
                            break
                        for b3 in range(b2 + 1, range_limit):
                            if time.time() - t_start > TIME_LIMIT - 2:
                                break
                            B = [b1, b2, b3]
                            
                            sums = [a + b for a in A for b in B]
                            
                            # Check distinctness first (cheap), then primality
                            if len(set(sums)) == 9:
                                if all(is_prime[s] for s in sums):
                                    return [A, B]
                                
    return best_result
# EVOLVE-BLOCK-END
