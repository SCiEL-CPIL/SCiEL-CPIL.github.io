"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def _factorize(n, primes):
    vec = []
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            vec.append((p, e))
    if n > 1:
        vec.append((n, 1))
    return vec

def _exponent_vector(n, primes):
    d = {}
    for p, e in _factorize(n, primes):
        d[p] = d.get(p, 0) + e
    return tuple(sorted(d.items()))

def _subset_products_distinct(nums, primes):
    """Check if all 2^k subset products are distinct using exponent vectors."""
    if not nums:
        return True
    # Map each number to its exponent vector (as a tuple of (prime, exp) pairs)
    vecs = []
    for n in nums:
        d = {}
        for p, e in _factorize(n, primes):
            d[p] = d.get(p, 0) + e
        # Convert to a canonical tuple: sorted list of (prime, exp)
        vecs.append(tuple(sorted(d.items())))
    
    # Now we need to check that all subset sums of these vectors are distinct.
    # A subset sum is a multiset of (prime, exp) pairs summed component-wise.
    # Represent a sum as a frozenset? No, it's a mapping from prime to total exp.
    # Better: represent as a tuple of (prime, total_exp) sorted by prime.
    
    # Start with the zero vector
    zero_vec = ()
    seen = {zero_vec}
    
    # For each new vector, generate all new subset sums by adding to existing ones
    for v in vecs:
        new_sums = set()
        for s in seen:
            # Add v to s
            # s is a tuple of (prime, exp), v is a tuple of (prime, exp)
            # We need to merge them
            d = dict(s)
            for p, e in v:
                d[p] = d.get(p, 0) + e
            new_s = tuple(sorted(d.items()))
            if new_s in seen:
                return False
            new_sums.add(new_s)
        seen.update(new_sums)
    return True

def _greedy_search(primes, deadline):
    """Greedy: pick numbers in order of 'rarity' to maximize count."""
    # All candidate numbers 2..40
    candidates = list(range(2, 41))
    # Precompute factorizations
    cand_vecs = {}
    for n in candidates:
        cand_vecs[n] = _exponent_vector(n, primes)
    
    selected = []
    selected_vecs = []
    used = set()
    
    # Try to add numbers that introduce new primes or new combinations
    # Strategy: sort by number of prime factors (fewer = more 'independent')
    # Actually, let's try a randomized greedy with multiple restarts
    
    best_found = 0
    best_set = []
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        random.shuffle(candidates)
        sel = []
        sel_vecs = []
        for n in candidates:
            if len(sel) >= 18:
                break
            if n in used:
                continue
            # Check if adding n keeps subset products distinct
            # We need to check: do any two subset products of sel become equal when we add n?
            # Equivalent: is there a subset S of sel such that product(S) == n * product(T) for some subset T of sel?
            # i.e., product(S) / product(T) == n
            # This means n must not be expressible as a ratio of two subset products of sel.
            # 
            # Simpler: just check all 2^(len(sel)+1) subset products directly.
            # For len(sel) <= 17, that's at most 2^18 = 262144 which is fine.
            test_set = sel + [n]
            if _subset_products_distinct(test_set, primes):
                sel.append(n)
                sel_vecs.append(cand_vecs[n])
        
        if len(sel) > len(best_set):
            best_set = sel[:]
            best_found = len(sel)
    
    return best_set, best_found

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s
    
    primes = _primes_upto(40)
    
    # First, try a deterministic approach: use primes and prime powers
    # Primes <= 40: 2,3,5,7,11,13,17,19,23,29,31,37 (12 primes)
    # Prime squares: 4,9,25,49 -> 49>40 so 4,9,25 (3 more)
    # That gives 15. Need 3 more.
    # 
    # Let's use the greedy randomized search
    best_set, best_found = _greedy_search(primes, deadline)
    
    # Try to improve by swapping
    if time.time() < deadline - 5.0:
        # Local search: try swapping one element for another
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current = best_set[:]
            for i in range(len(current)):
                if time.time() > deadline - 2.0:
                    break
                for n in range(2, 41):
                    if n in current:
                        continue
                    test = current[:]
                    test[i] = n
                    test.sort()
                    if _subset_products_distinct(test, primes):
                        # Check if this actually helps (it doesn't change size, but might enable future additions)
                        # For now, just accept if it's valid
                        if len(test) == len(current):
                            # Try to add more after this swap
                            for m in range(2, 41):
                                if m in test:
                                    continue
                                if len(test) >= 18:
                                    break
                                test2 = test + [m]
                                if _subset_products_distinct(test2, primes):
                                    if len(test2) > len(best_set):
                                        best_set = test2
                                        improved = True
                                    break
                        break
    
    # Ensure we return a valid list of ints
    result = list(best_set)
    # Verify once more
    if not _subset_products_distinct(result, primes):
        # Fallback: just primes
        result = [p for p in primes if p <= 40]
    
    return result
# EVOLVE-BLOCK-END
