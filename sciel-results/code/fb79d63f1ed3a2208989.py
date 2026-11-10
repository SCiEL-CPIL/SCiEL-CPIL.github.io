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
    if len(distinct) != len(sums):
        # duplicates are allowed but we count distinct
        pass
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
    deadline = start_time + 80  # 80s safety margin
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _score(best_A, best_B)
    
    # Precompute: we want sums to be pairwise coprime.
    # Sums range from 2 to 256.
    # Key constraint: no two distinct sums can share a factor > 1.
    
    # Strategy: search over candidate sets
    # For efficiency, note that if we use small numbers, sums are small
    # and more likely to share factors.
    
    # Let's do a randomized + local search approach
    import random
    random.seed(42)
    
    # First, try some structured approaches
    # Approach 1: Use numbers that make sums be primes or powers of distinct primes
    
    # Let's try: make all 9 sums be distinct primes (or 1)
    # a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3
    # These need to be pairwise coprime.
    
    # Try to find A, B such that sums are all distinct and pairwise coprime.
    
    # Randomized search with local improvement
    def random_candidate():
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        return sorted(A), sorted(B)
    
    def neighbor(A, B):
        """Generate a neighbor by changing one element."""
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
    
    # Try many random restarts with local search
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Random restart
        A, B = random_candidate()
        s = _score(A, B)
        
        if s > best_score:
            best_score = s
            best_A, best_B = A, B
            if best_score >= 9:
                break
        
        # Local search
        for _ in range(200):
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
                    if best_score >= 9:
                        return best_A, best_B
        
        if best_score >= 9:
            break
    
    return best_A, best_B
# EVOLVE-BLOCK-END
