import itertools
import math
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    
    # Max possible distinct sums is 9.
    # We aim for 9.
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_valid(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        if len(distinct_sums) < 9:
            return 0
        
        # Check pairwise coprimality
        for i in range(9):
            for j in range(i + 1, 9):
                if math.gcd(distinct_sums[i], distinct_sums[j]) > 1:
                    return 0
        return 9

    # Precomputed known valid solution if found
    # Let's try to find one quickly via random search in the function
    # Or use a deterministic search that is fast
    
    # Deterministic search in small range
    # A and B should be from small numbers to keep sums small (more primes)
    # We try A from odds, B from evens to ensure sums are odd (except 2, but 2 is even so sums are odd)
    # If A has odds and B has evens, sums are odd.
    # 1 is coprime to everything.
    
    t_start = time.time()
    TIME_LIMIT = 85
    
    # Candidate pools
    odds = [i for i in range(1, 129, 2)]
    evens = [i for i in range(2, 129, 2)]
    
    # Try small combinations first
    # A from first 10 odds, B from first 10 evens
    small_odds = odds[:10]
    small_evens = evens[:10]
    
    # Iterate over combinations
    # To make it faster, we can break early
    for A in itertools.combinations(small_odds, 3):
        if time.time() - t_start > TIME_LIMIT:
            break
        A = list(A)
        for B in itertools.combinations(small_evens, 3):
            if time.time() - t_start > TIME_LIMIT:
                break
            B = list(B)
            score = is_valid(A, B)
            if score == 9:
                return [A, B]
            
    # If not found in small range, expand or randomize
    # Random search
    rng = random.Random(42)
    attempts = 0
    while time.time() - t_start < TIME_LIMIT and attempts < 10000:
        attempts += 1
        A = rng.sample(odds, 3)
        B = rng.sample(evens, 3)
        score = is_valid(A, B)
        if score == 9:
            return [A, B]
            
    # Fallback to seed
    return [[1, 7, 13], [4, 10, 16]]
# EVOLVE-BLOCK-END
