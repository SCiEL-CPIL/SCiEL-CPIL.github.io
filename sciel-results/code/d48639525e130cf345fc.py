"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import math
import time

def _is_coprime_pairwise(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    distinct = list(set(vals))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _evaluate(A, B):
    """Return the number of distinct values in A+B if they are pairwise coprime, else 0."""
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    if _is_coprime_pairwise(distinct):
        return len(distinct)
    return 0

def construct():
    """
    Find A, B (each 3 distinct ints in [1,128]) maximizing |A+B| distinct values
    that are pairwise coprime.
    """
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Try the seed
    seed_A = [1, 7, 13]
    seed_B = [4, 10, 16]
    seed_score = _evaluate(seed_A, seed_B)
    if seed_score > best_score:
        best_score = seed_score
        best_A, best_B = seed_A[:], seed_B[:]
    
    # Candidate pool: Primes and 1, up to 100
    # Using primes increases the chance of coprime sums
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    candidates = [1] + primes[:20]
    candidates = sorted(set(candidates))
    
    # Search over combinations of A and B from candidates
    # To keep it fast, we limit the pool size or break on time
    # 20 candidates -> C(20,3) = 1140 combinations for A, 1140 for B.
    # 1140 * 1140 = 1.3 million pairs.
    # Each pair evaluation is ~100 operations.
    # 1.3M * 100 = 130M operations. Might take ~10-20s in Python.
    
    combs_A = list(itertools.combinations(candidates, 3))
    combs_B = list(itertools.combinations(candidates, 3))
    
    # Optimize: Iterate A, then B
    for A in combs_A:
        if time.time() > deadline:
            break
        for B in combs_B:
            if time.time() > deadline:
                break
            
            # Quick check: Are all sums distinct?
            sums = [a + b for a in A for b in B]
            if len(set(sums)) < 9:
                continue
            
            # Check pairwise coprime
            if _is_coprime_pairwise(sums):
                score = len(set(sums))
                if score > best_score:
                    best_score = score
                    best_A = list(A)
                    best_B = list(B)
                    if best_score == 9:
                        return [best_A, best_B]
    
    # If not found in structured search, try random search with broader range
    if best_score < 9 and time.time() < deadline:
        import random
        random.seed(42)
        for _ in range(50000):
            if time.time() > deadline:
                break
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            
            sums = [a + b for a in A for b in B]
            if len(set(sums)) < 9:
                continue
                
            if _is_coprime_pairwise(sums):
                score = len(set(sums))
                if score > best_score:
                    best_score = score
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return [best_A, best_B]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
