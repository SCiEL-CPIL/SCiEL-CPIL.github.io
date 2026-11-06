"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    """Check if all distinct values in vals are pairwise coprime."""
    vs = list(set(vals))
    for i in range(len(vs)):
        for j in range(i+1, len(vs)):
            if _gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _eval(A, B):
    """Return number of distinct sums if pairwise coprime, else 0."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) <= 1:
        return len(sums)
    if _pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def _search():
    """Systematic search for optimal A, B."""
    best = 0
    best_AB = ([1, 7, 13], [4, 10, 16])
    
    # Precompute: for each pair of values, check if they can coexist in a sumset
    # We want to maximize distinct sums that are pairwise coprime
    
    # Strategy: iterate over all possible A and B (3 distinct from 1..128)
    # That's C(128,3)^2 which is too large. Let's be smarter.
    
    # Key insight: sums range from 2 to 256. We need pairwise coprime sums.
    # Max pairwise coprime set in [2,256]: primes are pairwise coprime.
    # There are 54 primes in [2,256]. So 9 distinct prime sums is very feasible.
    
    # Let's try to find A, B such that all 9 sums are distinct and pairwise coprime.
    # Or at least as many as possible.
    
    # Brute force with pruning: 
    # For each A (3 distinct from 1..128), for each B (3 distinct from 1..128),
    # check if sums are pairwise coprime.
    
    # C(128,3) = 341,888. So total pairs ~ 1.17e11. Too many for full brute force.
    
    # Better approach: fix A, then for each B check. Still too many.
    
    # Alternative: use the fact that we need pairwise coprime sums.
    # Let's try random search + local improvement.
    
    import random
    import time
    
    deadline = time.time() + 50  # 50 second budget
    
    # Start with known good solution
    best_score = _eval([1, 7, 13], [4, 10, 16])
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Try structured search: use small numbers and primes
    # Sums should be pairwise coprime. If all sums are primes, that works.
    
    # Let's try: A = {a1, a2, a3}, B = {b1, b2, b3}
    # Sums: ai + bj for all i,j
    # We want these 9 sums to be pairwise coprime (ideally all prime and distinct)
    
    # Try systematic approach with small ranges first
    # Then expand
    
    # Phase 1: Try all A from small range, all B from small range
    max_range = 30
    candidates_a = list(range(1, max_range + 1))
    
    # Precompute all 3-combinations
    all_a = list(combinations(candidates_a, 3))
    all_b = list(combinations(candidates_a, 3))
    
    # This is 4060 * 4060 = ~16.5 million. Might be slow in Python.
    # Let's be smarter: for each A, compute what B values would give good sums.
    
    # Actually, let's just do a targeted search.
    # Key: if we pick A and B such that all sums are prime, we're done.
    
    # Let me try a different approach: enumerate possible sum sets.
    # A pairwise coprime set of size 9 in [2, 256].
    # Most such sets will consist of primes (and maybe 1, but sums >= 2).
    
    # Let's just do random restarts with local search.
    
    rng = random.Random(42)
    
    for restart in range(200000):
        if time.time() > deadline:
            break
        
        # Random A and B
        A = list(rng.sample(range(1, 129), 3))
        B = list(rng.sample(range(1, 129), 3))
        
        score = _eval(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                break
    
    # Local improvement
    if best_score < 9:
        for _ in range(100000):
            if time.time() > deadline:
                break
            # Mutate one element of A or B
            if rng.random() < 0.5:
                idx = rng.randint(0, 2)
                new_val = rng.randint(1, 128)
                if new_val not in best_A:
                    old = best_A[idx]
                    best_A[idx] = new_val
                    score = _eval(best_A, best_B)
                    if score >= best_score:
                        best_score = score
                        if best_score >= 9:
                            break
                    else:
                        best_A[idx] = old
            else:
                idx = rng.randint(0, 2)
                new_val = rng.randint(1, 128)
                if new_val not in best_B:
                    old = best_B[idx]
                    best_B[idx] = new_val
                    score = _eval(best_A, best_B)
                    if score >= best_score:
                        best_score = score
                        if best_score >= 9:
                            break
                    else:
                        best_B[idx] = old
    
    return best_A, best_B

# Pre-compute the best answer at module load time
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END
