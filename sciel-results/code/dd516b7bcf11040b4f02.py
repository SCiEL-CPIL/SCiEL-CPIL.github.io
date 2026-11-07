"""Rewritten program: maximize number of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute primes in 2..40
def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

PRIMES = [a for a in range(2, 41) if _is_prime(a)]
# Primes: 2,3,5,7,11,13,17,19,23,29,31,37 -> 12 primes

# Precompute prime exponent vectors for all numbers 2..40
def _prime_exponent_vector(n):
    vec = [0] * len(PRIMES)
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

EXP_VECTORS = {a: _prime_exponent_vector(a) for a in range(2, 41)}
NUM_PRIMES = len(PRIMES)

def _check_distinct_subset_products_fast(S):
    """
    Check if all 2^k subset products of S are distinct using prime exponent vectors.
    Two subset products are equal iff their prime exponent vectors sum to the same vector.
    """
    k = len(S)
    if k == 0:
        return True
    subset_sums = {tuple([0] * NUM_PRIMES)}
    for x in S:
        v = EXP_VECTORS[x]
        new_sums = set()
        for s in subset_sums:
            ns = tuple(s[j] + v[j] for j in range(NUM_PRIMES))
            if ns in subset_sums:
                return False
            new_sums.add(ns)
        subset_sums.update(new_sums)
    return True

def _check_add(S, x):
    """Check if adding x to S keeps subset products distinct. Incremental check."""
    k = len(S)
    if k == 0:
        return True
    subset_sums = {tuple([0] * NUM_PRIMES)}
    for i in range(k):
        v = EXP_VECTORS[S[i]]
        new_sums = set()
        for s in subset_sums:
            ns = tuple(s[j] + v[j] for j in range(NUM_PRIMES))
            new_sums.add(ns)
        subset_sums.update(new_sums)
    # Now check if adding x creates a collision
    v = EXP_VECTORS[x]
    for s in subset_sums:
        ns = tuple(s[j] + v[j] for j in range(NUM_PRIMES))
        if ns in subset_sums:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s limit
    
    # Start with primes (12 elements)
    base = list(PRIMES)
    
    # Verify base is valid
    if not _check_distinct_subset_products_fast(base):
        return base
    
    # Composites sorted by number of prime factors (simpler ones first)
    composites = [a for a in range(2, 41) if not _is_prime(a)]
    
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    composites.sort(key=lambda x: (prime_factor_count(x), x))
    
    best = list(base)
    
    # Greedy: try to add each composite if it keeps the set valid
    for c in composites:
        if _check_add(best, c):
            best.append(c)
    
    # Randomized restarts with different orderings
    import random
    random.seed(42)
    
    for trial in range(500):
        if time.time() > deadline:
            break
        trial_comps = list(composites)
        random.shuffle(trial_comps)
        trial_best = list(base)
        for c in trial_comps:
            if _check_add(trial_best, c):
                trial_best.append(c)
        if len(trial_best) > len(best):
            best = trial_best
    
    # Local search: try swapping one element for another
    # More targeted: for each element in best, try removing it and adding a better composite
    for trial in range(200):
        if time.time() > deadline:
            break
        current = list(best)
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                without_i = [x for j, x in enumerate(current) if j != i]
                for c in composites:
                    if c in current:
                        continue
                    candidate = without_i + [c]
                    if len(candidate) == len(current):
                        if _check_distinct_subset_products_fast(candidate):
                            current = candidate
                            improved = True
                            break
                if improved:
                    break
            if time.time() > deadline:
                break
        if len(current) > len(best):
            best = current
    
    # Try to add more elements after swaps
    for c in composites:
        if c not in best and _check_add(best, c):
            best.append(c)
    
    # Final: ensure we return Python ints in a list
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
