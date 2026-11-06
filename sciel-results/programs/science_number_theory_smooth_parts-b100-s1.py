"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def extract_small_factor(m):
        """Extract the complete factor of m composed of primes strictly below 24."""
        factor = 1
        for p in primes:
            while m % p == 0:
                factor *= p
                m //= p
        return factor
    
    def score_candidate(n):
        """Count distinct small-prime factors across n+1 to n+24."""
        distinct_factors = set()
        for i in range(1, 25):
            f = extract_small_factor(n + i)
            distinct_factors.add(f)
        return len(distinct_factors)
    
    # Search for minimum
    best_n = 1000
    best_score = score_candidate(best_n)
    
    # Check time budget
    if time.time() > deadline:
        return best_n
    
    # Search in chunks to allow time checking
    # Start from 1000, go up to 1000000
    # We'll search in steps, checking time periodically
    
    n = 1000
    chunk_size = 10000
    
    while n + chunk_size <= 1000000:
        if time.time() > deadline:
            break
        
        for i in range(n, n + chunk_size):
            s = score_candidate(i)
            if s < best_score:
                best_score = s
                best_n = i
                # If we found a very good score, we can potentially stop early
                # but let's continue to see if we can do better
                if best_score <= 2:
                    # This is likely optimal, but let's verify a bit more
                    pass
        
        n += chunk_size
    
    # Finish remaining
    while n <= 1000000:
        if time.time() > deadline:
            break
        s = score_candidate(n)
        if s < best_score:
            best_score = s
            best_n = n
        n += 1
    
    return best_n
# EVOLVE-BLOCK-END
