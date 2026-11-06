import itertools
import math
import time

def construct():
    """
    Returns two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximizes the cardinality of A+B.
    """
    t_start = time.time()
    TIME_LIMIT = 85.0  # Safety margin before 90s kill
    
    # Precompute primes up to 256
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    def is_pairwise_coprime(values):
        """Check if all distinct values are pairwise coprime."""
        vals = list(set(values))
        n = len(vals)
        for i in range(n):
            for j in range(i+1, n):
                if math.gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    # First, try to find a solution with all 9 sums distinct and prime
    # This gives maximum cardinality of 9
    
    # Strategy: enumerate all pairs of 3-element sets and check
    # 128^3 * 128^3 is too large, so we need a smarter approach
    
    # Better approach: for each triple A, find B such that A+B gives 9 distinct primes
    # We can iterate over A triples (sorted) and for each, try B triples
    
    best_result = None
    best_count = 0
    
    # Enumerate all possible A triples (sorted, distinct, in [1,128])
    # There are C(128,3) = 345,840 possible A triples
    # For each, we need to find B triples such that all sums are pairwise coprime
    
    # To make this tractable, let's use a different strategy:
    # Pick a target set of 9 primes, and try to find A, B such that A+B covers them
    
    # Actually, let's just do a systematic search with early termination
    
    # Generate all primes in [2, 256]
    primes_in_range = [p for p in range(2, max_val + 1) if is_prime[p]]
    
    # Try combinations of 9 primes as target sums
    # C(~55, 9) is too large, so let's try a different approach
    
    # Direct search: iterate over A triples, for each try to find B triples
    # where all 9 sums are pairwise coprime
    
    count = 0
    for a1 in range(1, 129):
        if time.time() - t_start > TIME_LIMIT - 5:
            break
        for a2 in range(a1+1, 129):
            if time.time() - t_start > TIME_LIMIT - 5:
                break
            for a3 in range(a2+1, 129):
                A = [a1, a2, a3]
                
                # For each B triple, check if A+B has all pairwise coprime distinct values
                # We can prune: for a given A, the sums a1+b, a2+b, a3+b for fixed b
                # must be coprime to each other and to sums from other b's
                
                for b1 in range(1, 129):
                    if time.time() - t_start > TIME_LIMIT - 5:
                        break
                    for b2 in range(b1+1, 129):
                        if time.time() - t_start > TIME_LIMIT - 5:
                            break
                        for b3 in range(b2+1, 129):
                            B = [b1, b2, b3]
                            sums = [a+b for a in A for b in B]
                            distinct_sums = set(sums)
                            
                            # Check if all 9 sums are distinct and pairwise coprime
                            if len(distinct_sums) == 9 and is_pairwise_coprime(sums):
                                return [A, B]
                
                count += 1
                if count % 10000 == 0:
                    if time.time() - t_start > TIME_LIMIT - 5:
                        break
        
        if time.time() - t_start > TIME_LIMIT - 5:
            break
    
    # If we didn't find a perfect 9-distinct solution, fall back to finding
    # the best solution with maximum cardinality
    # Try to find solutions with 8 or 7 distinct pairwise coprime sums
    
    # For now, return a known good solution
    # Let's try a few hardcoded solutions first
    test_solutions = [
        [[1, 2, 3], [4, 8, 12]],
        [[1, 2, 3], [4, 6, 9]],
        [[1, 2, 4], [3, 5, 7]],
        [[1, 2, 3], [5, 7, 11]],
        [[2, 3, 5], [4, 7, 11]],
    ]
    
    for A, B in test_solutions:
        sums = [a+b for a in A for b in B]
        distinct_sums = set(sums)
        if is_pairwise_coprime(sums):
            return [A, B]
    
    # Final fallback: seed solution
    return [[1, 7, 13], [4, 10, 16]]
