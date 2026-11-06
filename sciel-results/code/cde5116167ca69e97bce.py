"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime."""
    distinct = list(set(values))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _score_pair(A, B):
    """Return number of distinct pairwise-coprime sums, or -1 if not valid."""
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    if not _is_pairwise_coprime(distinct):
        return -1
    return len(distinct)

def construct():
    deadline = time.time() + 80  # 80s out of 90s budget
    best_count = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Verify the seed is valid
    seed_sums = [a + b for a in [1, 7, 13] for b in [4, 10, 16]]
    seed_distinct = list(set(seed_sums))
    if _is_pairwise_coprime(seed_distinct):
        best_count = len(seed_distinct)
    
    # Precompute: we want to maximize distinct sums that are pairwise coprime
    # Max possible is 9 (all a_i + b_j distinct)
    
    # Strategy 1: Random search over 3-element subsets
    # Range [1, 128]
    rng = random.Random(42)
    
    # For speed, precompute a list of all 3-element subsets is too large (128 choose 3 ~ 340k)
    # Instead, sample randomly
    
    # Also try structured approaches:
    # If A = {a, a+d, a+2d} and B = {b, b+d, b+2d}, sums are:
    # a+b, a+b+d, a+b+2d, a+d+b, a+d+b+d, a+d+b+2d, a+2d+b, a+2d+b+d, a+2d+b+2d
    # = a+b+kd for k in 0..4, with multiplicities 1,2,3,2,1
    # So distinct sums are 5 values: a+b, a+b+d, a+b+2d, a+b+3d, a+b+4d
    # These need to be pairwise coprime.
    
    # Try AP pairs
    for a in range(1, 127):
        for d in range(1, 40):
            A = [a, a + d, a + 2 * d]
            if A[2] > 128:
                break
            for b in range(1, 127):
                B = [b, b + d, b + 2 * d]
                if B[2] > 128:
                    break
                s = _score_pair(A, B)
                if s > best_count:
                    best_count = s
                    best_A = A[:]
                    best_B = B[:]
                if time.time() > deadline:
                    return [best_A, best_B]
    
    # Try AP with different differences
    for a in range(1, 127):
        for d1 in range(1, 30):
            A = [a, a + d1, a + 2 * d1]
            if A[2] > 128:
                continue
            for b in range(1, 127):
                for d2 in range(1, 30):
                    B = [b, b + d2, b + 2 * d2]
                    if B[2] > 128:
                        continue
                    s = _score_pair(A, B)
                    if s > best_count:
                        best_count = s
                        best_A = A[:]
                        best_B = B[:]
                    if time.time() > deadline:
                        return [best_A, best_B]
    
    # Random search
    for _ in range(500000):
        if time.time() > deadline:
            break
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = _score_pair(A, B)
        if s > best_count:
            best_count = s
            best_A = A[:]
            best_B = B[:]
            if best_count >= 9:
                break
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
