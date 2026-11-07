"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt, log2
from functools import lru_cache

# Deadline guard: kill after 85s wall-clock (5s safety margin under 90s limit)
_DEADLINE = time.time() + 85.0
def _time_up():
    return time.time() > _DEADLINE

# Candidate pool: 2..40
_CANDIDATES = list(range(2, 41))

# Precompute prime factorization for each candidate (as dict of prime->exponent)
def _factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

_FACTORS = {x: _factorize(x) for x in _CANDIDATES}

# Precompute: for each candidate, which other candidates share a prime factor (conflict set)
# Two numbers conflict if they share a prime factor, because then products can collide.
# Actually, the condition for distinct subset products is stronger: the set must be "multiplicatively independent".
# A sufficient (and necessary for integers > 1) condition is that no subset product equals another.
# For efficiency in search, we use a heuristic: numbers with shared prime factors are more likely to cause collisions.

def _can_add_fast(prods_set, x):
    """Check if adding x to the set keeps all subset products distinct.
    prods_set is the set of all current subset products.
    We need to check that for every p in prods_set, p*x is not already in prods_set,
    AND that x itself is not in prods_set (which is covered by p=1).
    """
    # For each existing product p, check if p*x collides with an existing product
    # Instead of computing all p*x, check: does there exist p1, p2 in prods_set such that p1*x = p2?
    # i.e., p2 / x is in prods_set and p2 % x == 0
    # But also p1*x must not equal any existing product, meaning:
    # For all p in prods_set: p*x not in prods_set
    # Equivalently: for all q in prods_set: q % x == 0 and q//x in prods_set is False (except when q==x and 1 in prods_set, which is the p=1 case)
    # Actually simpler: just check p*x for all p. But that's |prods_set| multiplications.
    # Alternative: for each q in prods_set, if q % x == 0 and q//x in prods_set, then collision.
    # This is the same cost but avoids creating new large integers.
    for q in prods_set:
        if q % x == 0:
            if q // x in prods_set:
                return False
    return True

def _add_element(prods_set, x):
    """Add x to the set, returning the new set of subset products.
    New products are {p*x for p in prods_set} union {x} (but x = 1*x so it's included).
    """
    new_prods = set()
    for p in prods_set:
        new_prods.add(p * x)
    new_prods.update(prods_set)
    return new_prods

def _greedy_build():
    """Greedy: add candidates in a fixed order that tends to minimize conflicts.
    Try adding in order of increasing value, or by some heuristic.
    """
    prods = {1}
    result = []
    for x in _CANDIDATES:
        if _can_add_fast(prods, x):
            prods = _add_element(prods, x)
            result.append(x)
    return result

def _greedy_build_desc():
    """Greedy descending."""
    prods = {1}
    result = []
    for x in reversed(_CANDIDATES):
        if _can_add_fast(prods, x):
            prods = _add_element(prods, x)
            result.append(x)
    return result

def _random_greedy(rng):
    """Randomized greedy: shuffle candidates, add if possible."""
    candidates = list(_CANDIDATES)
    rng.shuffle(candidates)
    prods = {1}
    result = []
    for x in candidates:
        if _can_add_fast(prods, x):
            prods = _add_element(prods, x)
            result.append(x)
    return result

def _local_search(start_set, rng, max_iters=5000):
    """Try to improve the set by swapping elements out and in.
    start_set: list of integers
    Returns improved list.
    """
    current = list(start_set)
    current_set = set(current)
    
    # Build initial prods
    prods = {1}
    for x in current:
        prods = _add_element(prods, x)
    
    best_len = len(current)
    best_set = list(current)
    
    for _ in range(max_iters):
        if _time_up():
            break
        
        # Try adding a random candidate not in current
        available = [x for x in _CANDIDATES if x not in current_set]
        if not available:
            break
        
        # Try to add one element
        added = False
        for x in available:
            if _can_add_fast(prods, x):
                prods = _add_element(prods, x)
                current.append(x)
                current_set.add(x)
                added = True
                if len(current) > best_len:
                    best_len = len(current)
                    best_set = list(current)
                break
        
        if not added:
            # Try swapping: remove one element, then try to add two different ones
            if len(current) < 2:
                break
            
            idx = rng.randrange(len(current))
            removed = current[idx]
            current.pop(idx)
            current_set.discard(removed)
            
            # Rebuild prods without removed element
            prods = {1}
            for x in current:
                prods = _add_element(prods, x)
            
            # Try to add two new elements
            added_count = 0
            tried = []
            for x in _CANDIDATES:
                if x in current_set or x == removed:
                    continue
                if _can_add_fast(prods, x):
                    prods = _add_element(prods, x)
                    current.append(x)
                    current_set.add(x)
                    added_count += 1
                    if len(current) > best_len:
                        best_len = len(current)
                        best_set = list(current)
                    if added_count >= 2:
                        break
            
            # If we couldn't add back at least 1, revert
            if added_count == 0:
                current.append(removed)
                current_set.add(removed)
                prods = _add_element(prods, removed)
    
    return best_set

def construct():
    """Construct a set of distinct integers in 2..40 with all 2^k subset products distinct.
    Maximize the number of elements.
    """
    start_time = time.time()
    
    # Try several strategies
    best = []
    
    # 1. Greedy ascending
    g1 = _greedy_build()
    if len(g1) > len(best):
        best = g1
    
    # 2. Greedy descending
    g2 = _greedy_build_desc()
    if len(g2) > len(best):
        best = g2
    
    # 3. Randomized greedy with local search
    rng = random.Random(42)
    
    # Do multiple random restarts
    for restart in range(200):
        if _time_up():
            break
        
        # Random greedy
        rg = _random_greedy(rng)
        if len(rg) > len(best):
            best = rg
        
        # Local search from this starting point
        ls = _local_search(rg, rng, max_iters=2000)
        if len(ls) > len(best):
            best = ls
        
        # Early exit if we hit the theoretical max (18)
        if len(best) >= 18:
            break
    
    # 4. More aggressive search: try to build from scratch with different orderings
    # Sort by number of prime factors (fewer factors = less likely to conflict)
    def prime_factor_count(x):
        return sum(_FACTORS[x].values())
    
    by_pf_count = sorted(_CANDIDATES, key=prime_factor_count)
    prods = {1}
    result = []
    for x in by_pf_count:
        if _can_add_fast(prods, x):
            prods = _add_element(prods, x)
            result.append(x)
    if len(result) > len(best):
        best = result
    
    # Local search from this
    if not _time_up():
        ls2 = _local_search(result, rng, max_iters=5000)
        if len(ls2) > len(best):
            best = ls2
    
    # Final verification
    # Verify the best set has all distinct subset products
    def _verify(s):
        prods = {1}
        for x in s:
            new_prods = set()
            for p in prods:
                new_prods.add(p * x)
            new_prods.update(prods)
            prods = new_prods
        return len(prods) == 2 ** len(s)
    
    if not _verify(best):
        # Fallback to a known valid small set
        best = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    return list(best)
# EVOLVE-BLOCK-END
