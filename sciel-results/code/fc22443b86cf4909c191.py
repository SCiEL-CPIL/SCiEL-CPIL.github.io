"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

def _exponent_vector(num, primes):
    v = []
    for p in primes:
        cnt = 0
        while num % p == 0:
            num //= p
            cnt += 1
        v.append(cnt)
    return tuple(v)

def _subset_products_distinct(nums, primes):
    k = len(nums)
    vectors = [_exponent_vector(x, primes) for x in nums]
    seen = set()
    seen.add((0,)*len(primes))
    for mask in range(1, 1 << k):
        vec = [0]*len(primes)
        for i in range(k):
            if mask & (1 << i):
                for j in range(len(primes)):
                    vec[j] += vectors[i][j]
        t = tuple(vec)
        if t in seen:
            return False
        seen.add(t)
    return True

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    # These 12 primes have distinct subset products
    result = primes[:]
    
    # Try to add more numbers
    candidates = [x for x in range(2, 41) if x not in result]
    
    # Try adding one at a time, preferring smaller composites
    for c in candidates:
        if _subset_products_distinct(result + [c], primes):
            result.append(c)
    
    # Try to swap: remove a prime, add a composite
    # Not likely to help but try a few swaps
    improved = True
    while improved:
        improved = False
        for i in range(len(result)):
            for c in candidates:
                if c in result:
                    continue
                new_result = result[:i] + result[i+1:] + [c]
                new_result.sort()
                if _subset_products_distinct(new_result, primes):
                    # Check if this actually helps (it shouldn't in terms of size, 
                    # but might allow further additions)
                    # For now, just try to extend from here
                    pass
    
    # More systematic: try all pairs of removals and additions
    # This is too slow, so let's just do greedy with backtracking
    
    # Actually, let's do a proper search for size > 12
    # Try to find a set of size 13
    # We'll use a recursive search with pruning
    
    best = result
    
    # Try to add one more element by checking all candidates
    # against the current best
    # Then try to add two more by checking pairs, etc.
    
    # For efficiency, let's try adding up to 2 more elements
    for c1 in candidates:
        if c1 in result:
            continue
        temp = result + [c1]
        if _subset_products_distinct(temp, primes):
            if len(temp) > len(best):
                best = temp[:]
            for c2 in candidates:
                if c2 in temp:
                    continue
                temp2 = temp + [c2]
                if _subset_products_distinct(temp2, primes):
                    if len(temp2) > len(best):
                        best = temp2[:]
    
    # Ensure all are Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
