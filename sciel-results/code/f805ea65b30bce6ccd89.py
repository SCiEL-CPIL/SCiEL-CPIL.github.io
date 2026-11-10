"""Search-based program: find A,B with 3 distinct integers in [1,128] maximizing |A+B| with pairwise coprime sums."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin under 90s wall-clock
    
    def distinct_sums(A, B):
        return {a + b for a in A for b in B}
    
    def is_pairwise_coprime(vals):
        lst = list(vals)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if gcd(lst[i], lst[j]) != 1:
                    return False
        return True
    
    def score_of(A, B):
        s = distinct_sums(A, B)
        if len(s) < 2:
            return -1
        if not is_pairwise_coprime(s):
            return -1
        return len(s)
    
    # Start with the seed solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_of(best_A, best_B)
    
    # Systematic search: try to find configurations with more than 9 distinct sums
    # Key insight: to get 9 distinct sums, all a_i + b_j must be different
    # This requires that (a2-a1) != (b2-b1), (a2-a1) != (b3-b2), etc.
    # And all sums must be pairwise coprime
    
    # Strategy: Use sets with large gaps to ensure distinct sums
    # Then filter for pairwise coprime property
    
    # Precompute: for each pair of distinct values in [1,128], check if their sum is coprime-friendly
    # We'll search over structured families first, then random
    
    # Family 1: Arithmetic progressions with different common differences
    # A = [a, a+d1, a+2*d1], B = [b, b+d2, b+2*d2]
    # Sums: a+b, a+b+d2, a+b+2*d2, a+d1+b, a+d1+b+d2, a+d1+b+2*d2, a+2*d1+b, a+2*d1+b+d2, a+2*d1+b+2*d2
    # All distinct if d1 != d2 and d1, d2 > 0
    
    found_better = False
    for a in range(1, 100):
        if time.time() > deadline:
            break
        for d1 in range(1, 40):
            if time.time() > deadline:
                break
            a2 = a + d1
            a3 = a + 2 * d1
            if a3 > 128:
                break
            for b in range(1, 100):
                if time.time() > deadline:
                    break
                for d2 in range(1, 40):
                    if time.time() > deadline:
                        break
                    if d2 == d1:
                        continue
                    b2 = b + d2
                    b3 = b + 2 * d2
                    if b3 > 128:
                        break
                    A = [a, a2, a3]
                    B = [b, b2, b3]
                    s = score_of(A, B)
                    if s > best_score:
                        best_score = s
                        best_A = A[:]
                        best_B = B[:]
                        if s == 9:
                            found_better = True
                            # Try to find even better or confirm 9 is max
                            # Continue searching briefly for robustness
                            break
                if found_better and best_score == 9:
                    break
            if found_better and best_score == 9:
                break
        if found_better and best_score == 9:
            break
    
    # If we found 9, do a quick random search to see if we can verify or find edge cases
    if best_score >= 9:
        # Quick random verification
        import random
        random.seed(42)
        for _ in range(50000):
            if time.time() > deadline:
                break
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            s = score_of(A, B)
            if s > best_score:
                best_score = s
                best_A = A
                best_B = B
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
