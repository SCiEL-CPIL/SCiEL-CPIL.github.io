"""Rewritten program: search for large sets with distinct subset products."""
# EVOLVE-BLOCK-START
import time
import math
import random
from functools import lru_cache
from itertools import combinations

MAX_N = 40
MAX_SIZE = 18
DEADLINE_SECONDS = 75  # safety margin under 90s kill

# Precompute primes up to 40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NUM_PRIMES = len(_PRIMES)

# Precompute exponent vectors for each integer 2..40
_EXP_VECTORS = {}
for val in range(2, MAX_N + 1):
    vec = [0] * _NUM_PRIMES
    x = val
    for i, p in enumerate(_PRIMES):
        while x % p == 0:
            vec[i] += 1
            x //= p
    _EXP_VECTORS[val] = tuple(vec)

# Precompute a bitmask-friendly representation: use tuple of ints
# For subset product distinctness, we need all subset products to be distinct.
# This is equivalent to: the exponent vectors are linearly independent over the non-negative integers
# in the sense that no two subsets have the same sum of exponent vectors.
# A sufficient (and necessary) condition: the vectors are "sum-distinct", i.e., all 2^k subset sums are distinct.

def _subset_sums_distinct(vecs):
    """Check if all subset sums of the given exponent vectors are distinct.
    vecs: list of tuples of small ints.
    Returns True if all 2^len(vecs) subset sums are distinct."""
    k = len(vecs)
    if k == 0:
        return True
    # Use a set of tuples for subset sums
    sums = {tuple([0] * _NUM_PRIMES)}
    for v in vecs:
        new_sums = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(_NUM_PRIMES))
            new_sums.add(ns)
        sums = sums | new_sums
        if len(sums) < 2 ** (len(vecs)):  # early check: if we've added len(vecs) vectors, need 2^len(vecs) distinct
            # Actually after adding the j-th vector (0-indexed), we should have 2^(j+1) distinct sums
            pass
    # After processing all k vectors, we need exactly 2^k distinct sums
    return len(sums) == 2 ** k

def _check_set(vals):
    """Check if the set of values has distinct subset products."""
    vecs = [_EXP_VECTORS[v] for v in vals]
    return _subset_sums_distinct(vecs)

def _greedy_construct():
    """Greedy: try to build a large set with distinct subset products."""
    # Start with primes (they always work)
    result = list(_PRIMES)
    # Try to add composites in order of "ease" (smaller exponent vectors first)
    candidates = list(range(4, MAX_N + 1))
    # Sort candidates by sum of exponents (simpler vectors are easier to add)
    candidates.sort(key=lambda v: sum(_EXP_VECTORS[v]))
    
    for v in candidates:
        if v in result:
            continue
        test_set = result + [v]
        if _check_set(test_set):
            result.append(v)
        if len(result) >= MAX_SIZE:
            break
    return result

def _local_search(start_set, time_limit):
    """Local search: try to improve the set by swapping elements."""
    current = list(start_set)
    current_vecs = [_EXP_VECTORS[v] for v in current]
    best = current
    
    deadline = time.time() + time_limit
    candidates = [v for v in range(2, MAX_N + 1) if v not in current]
    
    while time.time() < deadline:
        improved = False
        # Try adding
        for v in candidates:
            if v in current:
                continue
            test = current + [v]
            if _check_set(test):
                current = test
                candidates = [x for x in candidates if x != v]
                improved = True
                if len(current) >= MAX_SIZE:
                    return current
        # Try swapping: remove one, add one
        if not improved and len(current) < MAX_SIZE:
            # Try removing a composite and adding a different composite
            for i, v in enumerate(current):
                if v in _PRIMES:
                    continue  # keep primes
                for w in candidates:
                    test = [x for j, x in enumerate(current) if j != i] + [w]
                    if _check_set(test):
                        current = test
                        candidates = [x for x in candidates if x != w]
                        if w in current:
                            candidates.remove(w)
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    return current

def _randomized_search(time_limit):
    """Randomized restarts with greedy + local search."""
    deadline = time.time() + time_limit
    best_size = 0
    best_set = []
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Randomly shuffle candidates and do greedy
        candidates = list(range(2, MAX_N + 1))
        rng.shuffle(candidates)
        
        result = []
        for v in candidates:
            if len(result) >= MAX_SIZE:
                break
            test = result + [v]
            if _check_set(test):
                result.append(v)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = list(result)
        
        # Local search from this result
        result = _local_search(result, 0.5)
        if len(result) > best_size:
            best_size = len(result)
            best_set = list(result)
        
        if best_size >= MAX_SIZE:
            break
    
    return best_set

def _structured_search(time_limit):
    """Structured search: prioritize primes and numbers with few prime factors."""
    deadline = time.time() + time_limit
    best_size = 0
    best_set = []
    
    # Always start with all primes
    base = list(_PRIMES)
    base_size = len(base)
    
    # Try adding composites in various orders
    composites = [v for v in range(2, MAX_N + 1) if v not in _PRIMES]
    
    # Sort by number of prime factors (with multiplicity)
    composites.sort(key=lambda v: sum(_EXP_VECTORS[v]))
    
    # Try different orderings
    orderings = [
        composites,
        sorted(composites, reverse=True),
        sorted(composites, key=lambda v: v),
        sorted(composites, key=lambda v: -v),
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        result = list(base)
        for v in ordering:
            if len(result) >= MAX_SIZE:
                break
            test = result + [v]
            if _check_set(test):
                result.append(v)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = list(result)
        
        # Local search
        result = _local_search(result, 1.0)
        if len(result) > best_size:
            best_size = len(result)
            best_set = list(result)
        
        if best_size >= MAX_SIZE:
            break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # First, try the structured search (primes + composites)
    best_set = _structured_search(30.0)
    
    # If we haven't hit max size, try randomized search
    if len(best_set) < MAX_SIZE and time.time() < deadline - 5:
        remaining = min(30.0, deadline - time.time() - 5)
        rand_set = _randomized_search(remaining)
        if len(rand_set) > len(best_set):
            best_set = rand_set
    
    # Final local search pass
    if time.time() < deadline - 2:
        remaining = min(10.0, deadline - time.time() - 2)
        final_set = _local_search(best_set, remaining)
        if len(final_set) > len(best_set):
            best_set = final_set
    
    # Verify the result
    if not _check_set(best_set):
        # Fallback to primes
        best_set = list(_PRIMES)
    
    # Ensure return type is list of Python ints
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
