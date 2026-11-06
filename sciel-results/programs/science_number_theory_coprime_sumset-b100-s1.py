"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import math
import time

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    TIME_LIMIT = 85  # safety margin before 90s kill
    
    # Precompute GCD helper
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_pairwise_coprime(sums):
        """Check if all distinct values in sums are pairwise coprime."""
        distinct = list(set(sums))
        n = len(distinct)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score(A, B):
        """Return cardinality of distinct A+B if pairwise coprime, else 0."""
        sums = [a + b for a in A for b in B]
        distinct = set(sums)
        if not is_pairwise_coprime(sums):
            return 0
        return len(distinct)
    
    # First, try to find a solution with all 9 distinct sums
    # All sums must be pairwise coprime. Easiest: all sums prime.
    
    # Precompute primes up to 256
    MAX_SUM = 256
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Check seed
    s = score(best_A, best_B)
    if s > best_score:
        best_score = s
    
    # Try to find all-9-distinct solutions where all sums are prime
    # This guarantees pairwise coprime
    # A = [a1, a2, a3], B = [b1, b2, b3]
    # All 9 sums a_i + b_j must be prime and distinct
    
    # Strategy: iterate over small values for A and B
    # To keep sums small and prime-friendly, use odd+even or even+odd patterns
    
    # Let's try: A has one even, two odds; B has one even, two odds
    # Or A all odd, B all even (or vice versa) to ensure sums are odd (more likely prime)
    
    # Try A with all odd, B with all even
    # Sums will be odd, which is good for primality
    
    # Iterate over possible A values (odd, 1-127) and B values (even, 2-128)
    # But 127*127*127*127 is too large. Need smarter search.
    
    # Smarter: fix A, then find B such that all sums are prime
    # For a given A = [a1, a2, a3], we need B = [b1, b2, b3] such that
    # a_i + b_j is prime for all i,j, and all 9 sums are distinct.
    
    # Let's try small A values first
    odd_vals = list(range(1, 128, 2))  # odd numbers 1 to 127
    even_vals = list(range(2, 129, 2))  # even numbers 2 to 128
    
    # Try combinations of A (odd) and B (even)
    # To limit search, try small values first
    max_search_val = 50  # limit initial search to small values for speed
    
    # Phase 1: Quick search with small values
    for a1 in range(1, min(128, max_search_val), 2):
        for a2 in range(a1+2, min(128, max_search_val), 2):
            for a3 in range(a2+2, min(128, max_search_val), 2):
                A = [a1, a2, a3]
                for b1 in range(2, min(129, max_search_val), 2):
                    for b2 in range(b1+2, min(129, max_search_val), 2):
                        for b3 in range(b2+2, min(129, max_search_val), 2):
                            B = [b1, b2, b3]
                            sums = [a + b for a in A for b in B]
                            if len(set(sums)) == 9:
                                if all(is_prime[s] for s in sums):
                                    best_score = 9
                                    best_A = A
                                    best_B = B
                                    return [best_A, best_B]
                            elif is_pairwise_coprime(sums):
                                sc = len(set(sums))
                                if sc > best_score:
                                    best_score = sc
                                    best_A = A
                                    best_B = B
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
            if time.time() - t_start > TIME_LIMIT - 5:
                break
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        if best_score == 9:
            break
        if time.time() - t_start > TIME_LIMIT - 5:
            break
    
    if time.time() - t_start > TIME_LIMIT - 3:
        return [best_A, best_B]
    
    # Phase 2: If we didn't find 9, try mixed parity and larger values
    # Also try A even, B odd
    if best_score < 9:
        for a1 in range(2, min(129, max_search_val), 2):
            for a2 in range(a1+2, min(129, max_search_val), 2):
                for a3 in range(a2+2, min(129, max_search_val), 2):
                    A = [a1, a2, a3]
                    for b1 in range(1, min(128, max_search_val), 2):
                        for b2 in range(b1+2, min(128, max_search_val), 2):
                            for b3 in range(b2+2, min(128, max_search_val), 2):
                                B = [b1, b2, b3]
                                sums = [a + b for a in A for b in B]
                                if len(set(sums)) == 9:
                                    if all(is_prime[s] for s in sums):
                                        best_score = 9
                                        best_A = A
                                        best_B = B
                                        return [best_A, best_B]
                                elif is_pairwise_coprime(sums):
                                    sc = len(set(sums))
                                    if sc > best_score:
                                        best_score = sc
                                        best_A = A
                                        best_B = B
                    if time.time() - t_start > TIME_LIMIT - 5:
                        break
                if time.time() - t_start > TIME_LIMIT - 5:
                    break
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            if best_score == 9:
                break
            if time.time() - t_start > TIME_LIMIT - 5:
                break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
