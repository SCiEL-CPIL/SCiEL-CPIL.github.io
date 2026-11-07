"""Rewritten program: maximize number of distinct integers in 2..40 with distinct subset products."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factor exponent vectors for each number in 2..40
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

# Build exponent vectors for all numbers 2..40
# Each vector is a tuple of exponents for primes [2,3,5,7,11,13,17,19,23,29,31,37]
def _exponent_vector(n):
    vec = [0] * len(PRIMES)
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    return tuple(vec)

EXP_VECTORS = {}
for a in range(2, 41):
    EXP_VECTORS[a] = _exponent_vector(a)

def _check_distinct_subset_products(S):
    """
    Check if all 2^k subset products of S are distinct using exponent vectors.
    All subset sums of the exponent vectors must be distinct.
    """
    if not S:
        return True
    # Use set of tuples for subset sums
    sums = {tuple([0] * len(PRIMES))}
    for x in S:
        v = EXP_VECTORS[x]
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(len(PRIMES)))
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def _try_add_vectors(sums, x):
    """
    Try to add x to the set of exponent vector sums.
    Returns None if collision, else set of new sums.
    """
    v = EXP_VECTORS[x]
    new_sums = set()
    for s in sums:
        ns = tuple(s[i] + v[i] for i in range(len(PRIMES)))
        if ns in sums or ns in new_sums:
            return None
        new_sums.add(ns)
    return new_sums

def _greedy_add(sums, candidates, rng, deadline):
    """
    Greedily add elements from candidates to sums, trying in random order.
    Returns updated sums and list of added elements.
    """
    added = []
    remaining = list(candidates)
    rng.shuffle(remaining)
    
    for c in remaining:
        if time.time() > deadline:
            break
        new_sums = _try_add_vectors(sums, c)
        if new_sums is not None:
            sums.update(new_sums)
            added.append(c)
    
    return sums, added

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit
    
    # Start with primes (12 elements) - always valid
    base = list(PRIMES)
    
    # Initialize sum set for the base
    best = list(base)
    best_sums = {tuple([0] * len(PRIMES))}
    for x in best:
        new_sums = _try_add_vectors(best_sums, x)
        if new_sums is not None:
            best_sums.update(new_sums)
    
    # Composites to try adding
    composites = [a for a in range(2, 41) if not _is_prime(a)]
    
    # Multiple randomized greedy restarts
    best_result = list(best)
    best_size = len(best)
    
    # Try several random orderings
    for trial in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(trial)
        
        # Start from primes
        current_sums = {tuple([0] * len(PRIMES))}
        current_set = list(PRIMES)
        
        for x in current_set:
            new_sums = _try_add_vectors(current_sums, x)
            if new_sums is None:
                # Should not happen for primes
                break
            current_sums.update(new_sums)
        
        # Try adding composites in random order
        composites_list = list(composites)
        rng.shuffle(composites_list)
        
        for c in composites_list:
            if time.time() > deadline:
                break
            new_sums = _try_add_vectors(current_sums, c)
            if new_sums is not None:
                current_sums.update(new_sums)
                current_set.append(c)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = list(current_set)
            if best_size >= 18:
                break  # Can't do better than 18
    
    # If we didn't find anything better than 12, return the primes
    if best_size <= len(PRIMES):
        return [int(x) for x in best]
    
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
