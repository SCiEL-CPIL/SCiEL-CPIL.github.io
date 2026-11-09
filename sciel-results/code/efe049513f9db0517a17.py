"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _is_pairwise_coprime(nums):
    """Check if all distinct values in nums are pairwise coprime."""
    distinct = list(set(nums))
    n = len(distinct)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute primes up to 256 (max sum is 128+128)
    max_val = 128
    max_sum = 2 * max_val
    
    # Generate all 3-element subsets of [1, 128]
    # That's C(128,3) = 341,376 subsets
    # For each A, try to find B such that A+B has max distinct pairwise-coprime sums
    
    # Strategy: iterate over all A, and for each A, iterate over all B,
    # compute A+B, check pairwise coprime, track best.
    # But C(128,3)^2 is too large (~1.1e11).
    
    # Better: iterate over all A, and for each A, iterate over all B,
    # but prune early. Or: iterate over all A, and for each A, find best B.
    
    # Actually, let's think about the maximum possible distinct sums.
    # |A| = 3, |B| = 3, so A+B has at most 9 distinct values.
    # We want all 9 to be pairwise coprime.
    
    # Let's try: for each A (341,376 options), for each B (341,376 options),
    # that's ~1.17e11 which is too slow.
    
    # Alternative: for each A, generate all 9 sums for each B.
    # We can precompute for each A which B's give pairwise-coprime sums.
    
    # Even better: iterate over A, and for each A, iterate over B,
    # but break early if we find a good solution.
    
    # Let's first try a smarter approach:
    # Precompute all possible sums (2..256) and which are prime.
    # For A+B to have pairwise coprime elements, each element must be
    # either 1 or prime (since if any element is composite, it shares a factor
    # with itself... wait, no. Pairwise coprime means gcd(a,b)=1 for a!=b.
    # A composite number can be coprime with others. E.g., 4 and 9 are coprime.
    # But 4 and 6 are not. So we need the set to be pairwise coprime.
    
    # Let's just do a direct search but with early termination.
    
    best_count = 0
    best_A = None
    best_B = None
    
    # To speed up, let's iterate over A and B but use the fact that
    # we can check pairwise coprime quickly.
    
    # C(128,3) = 341,376
    # For each A, we need to check against each B. That's 341,376 * 341,376 ~ 1.17e11.
    # Too slow.
    
    # Better approach: for each A, compute the 9 sums with each B, but
    # we can pre-filter B's based on the sums.
    
    # Actually, let's think differently. The sumset A+B has 9 elements (possibly with repeats).
    # We want the distinct elements to be pairwise coprime.
    
    # Let's try: iterate over all A, and for each A, iterate over all B,
    # but limit the number of B's we check per A to keep within time.
    
    # Or: use a random search with many iterations.
    
    # Let's try a hybrid: first do a structured search over "nice" A and B
    # (e.g., arithmetic progressions or sets with small gcd properties),
    # then do random search if time permits.
    
    # First, let's try all pairs where A and B are arithmetic progressions.
    # A = [a, a+d, a+2d], B = [b, b+e, b+2e]
    # Sums: a+b, a+b+e, a+b+2e, a+d+b, a+d+b+e, a+d+b+2e, a+2d+b, a+2d+b+e, a+2d+b+2e
    # = (a+b) + i*d + j*e for i,j in {0,1,2}
    
    # This is a 3x3 grid of sums. We want all distinct values to be pairwise coprime.
    
    # Let's search over a, d, b, e with a, a+d, a+2d in [1,128] and b, b+e, b+2e in [1,128].
    
    for a in range(1, 127):  # a+2d <= 128
        for d in range(1, 128):
            if a + 2*d > 128:
                break
            A = [a, a+d, a+2*d]
            for b in range(1, 127):
                for e in range(1, 128):
                    if b + 2*e > 128:
                        break
                    B = [b, b+e, b+2*e]
                    
                    # Compute sums
                    sums = set()
                    for x in A:
                        for y in B:
                            sums.add(x + y)
                    
                    if len(sums) > best_count:
                        if _is_pairwise_coprime(list(sums)):
                            best_count = len(sums)
                            best_A = A[:]
                            best_B = B[:]
                    
                    if time.time() > deadline:
                        if best_A is not None:
                            return [best_A, best_B]
                        return [[1, 7, 13], [4, 10, 16]]
    
    if best_A is not None:
        return [best_A, best_B]
    
    # If AP search didn't find anything good, try random search
    import random
    random.seed(42)
    
    for _ in range(1000000):
        if time.time() > deadline:
            break
        A = sorted(random.sample(range(1, 129), 3))
        B = sorted(random.sample(range(1, 129), 3))
        sums = set()
        for x in A:
            for y in B:
                sums.add(x + y)
        if len(sums) > best_count:
            if _is_pairwise_coprime(list(sums)):
                best_count = len(sums)
                best_A = A[:]
                best_B = B[:]
    
    if best_A is not None:
        return [best_A, best_B]
    
    return [[1, 7, 13], [4, 10, 16]]

# EVOLVE-BLOCK-END
