"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
import time

def construct():
    # Deterministic construction of 9 pairwise coprime distinct sums.
    # We use 3-term arithmetic progressions of primes with common difference 6.
    # APs: (5, 11, 17), (7, 13, 19), (31, 37, 43)
    # Let B = {1, 7, 13} (diff 6)
    # Then A must be {4, 6, 30} to generate these sums.
    A = [4, 6, 30]
    B = [1, 7, 13]
    
    # Verification (optional, for safety)
    sums = set(a + b for a in A for b in B)
    # Check pairwise coprime
    vals = sorted(sums)
    valid = True
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if gcd(vals[i], vals[j]) != 1:
                valid = False
                break
        if not valid:
            break
    
    if valid and len(sums) == 9:
        return [A, B]
    else:
        # Fallback to random search if construction fails (should not happen)
        import random
        random.seed(98765)
        best_A, best_B, best_count = [1, 7, 13], [4, 10, 16], 0
        for _ in range(1000):
            A_r = sorted(random.sample(range(1, 129), 3))
            B_r = sorted(random.sample(range(1, 129), 3))
            s = set(a + b for a in A_r for b in B_r)
            v = sorted(s)
            ok = all(gcd(v[i], v[j]) == 1 for i in range(len(v)) for j in range(i+1, len(v)))
            if ok and len(s) > best_count:
                best_count = len(s)
                best_A, best_B = A_r, B_r
                if best_count == 9:
                    break
        return [best_A, best_B]
# EVOLVE-BLOCK-END
