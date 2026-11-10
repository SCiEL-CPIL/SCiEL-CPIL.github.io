"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations, product

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _are_pairwise_coprime(nums):
    """Check if all distinct values in nums are pairwise coprime."""
    distinct = list(set(nums))
    n = len(distinct)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return number of distinct sums if pairwise coprime, else 0."""
    sums = [a + b for a in A for b in B]
    distinct = set(sums)
    # Check pairwise coprimality of distinct sums
    dlist = list(distinct)
    n = len(dlist)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(dlist[i], dlist[j]) != 1:
                return 0
    return len(distinct)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Known good solution if found during development or hardcoded
    # Let's try to find one via a fast search first.
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _score(best_A, best_B)
    
    # If the initial guess is already 9, return immediately.
    if best_score == 9:
        return best_A, best_B

    # Fast deterministic search over a subset of promising candidates
    # We focus on small numbers and numbers with prime properties.
    # We limit the search space to avoid timeout.
    
    # Strategy: Iterate over A from a small set, and B from a small set.
    # If we find 9, we stop.
    
    # Candidate pool: 1..40 is likely sufficient for small distinct sums.
    # We can expand if needed, but 9 distinct sums in [2, 80] is tight.
    
    candidates = list(range(1, 41))
    
    # To speed up, we can pre-filter B for a given A.
    # But full product of C(40,3)^2 is ~ 10^6 * 10^6 = 10^12 (too big).
    # We need a smarter search.
    
    # Heuristic: Randomized search with a tighter time box and better initialization.
    import random
    random.seed(42)
    
    def random_candidate():
        # Bias towards smaller numbers?
        # Or mix of small and large?
        # Let's try uniform first.
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        return sorted(A), sorted(B)
    
    def neighbor(A, B):
        A2 = list(A)
        B2 = list(B)
        if random.random() < 0.5:
            idx = random.randint(0, 2)
            A2[idx] = random.randint(1, 128)
            if len(set(A2)) < 3:
                return None
        else:
            idx = random.randint(0, 2)
            B2[idx] = random.randint(1, 128)
            if len(set(B2)) < 3:
                return None
        return sorted(A2), sorted(B2)
    
    # Run a more aggressive search
    # 1. Try 1000 random restarts with short local search
    # 2. If not found, try structured search (arithmetic progressions)
    
    # Structured Search:
    # A = [a, a+d, a+2d], B = [b, b+e, b+2e]
    # Sums are a+b + (i*d + j*e) for i,j in 0..2.
    # We need these 9 values to be pairwise coprime.
    
    found_9 = False
    max_structured_iters = 5000
    
    for a in range(1, 20):
        for d in range(1, 10):
            A = [a, a+d, a+2*d]
            if A[2] > 128: continue
            for b in range(1, 20):
                for e in range(1, 10):
                    B = [b, b+e, b+2*e]
                    if B[2] > 128: continue
                    s = _score(A, B)
                    if s > best_score:
                        best_score = s
                        best_A, best_B = A, B
                        if s == 9:
                            return best_A, best_B
                    if time.time() > deadline:
                        return best_A, best_B
                        
    # If structured search didn't find 9, do random search
    if best_score < 9:
        iterations = 0
        while time.time() < deadline:
            iterations += 1
            A, B = random_candidate()
            s = _score(A, B)
            if s > best_score:
                best_score = s
                best_A, best_B = A, B
                if best_score == 9:
                    return best_A, best_B
            
            # Local search
            for _ in range(50):
                if time.time() > deadline:
                    break
                nb = neighbor(A, B)
                if nb is None:
                    continue
                A2, B2 = nb
                s2 = _score(A2, B2)
                if s2 >= s:
                    A, B = A2, B2
                    s = s2
                    if s > best_score:
                        best_score = s
                        best_A, best_B = A, B
                        if best_score == 9:
                            return best_A, best_B
    
    return best_A, best_B
# EVOLVE-BLOCK-END
