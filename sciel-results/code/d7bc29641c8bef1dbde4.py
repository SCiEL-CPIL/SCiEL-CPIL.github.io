"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _prime_factorization(n):
    """Return dict of prime -> exponent."""
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

def _subset_products_distinct(selected):
    """Check all 2^k subset products are distinct."""
    current = {1}
    for x in selected:
        new_prods = {p * x for p in current}
        if len(new_prods) != len(current):
            return False
        if current & new_prods:
            return False
        current |= new_prods
    return True

def _greedy_add(candidates_order):
    """Greedily add candidates in given order if no collision."""
    selected = []
    current = {1}
    for x in candidates_order:
        new_prods = {p * x for p in current}
        if len(new_prods) == len(current) and not (current & new_prods):
            selected.append(x)
            current |= new_prods
    return selected

def construct():
    start = time.time()
    timeout = 85.0
    
    candidates = list(range(2, 41))
    
    # Generate multiple orderings and try each
    orderings = []
    
    # Ordering 1: ascending
    orderings.append(candidates[:])
    # Ordering 2: descending
    orderings.append(candidates[::-1])
    # Ordering 3: primes first, then composites
    primes = [x for x in candidates if _is_prime(x)]
    composites = [x for x in candidates if not _is_prime(x)]
    orderings.append(primes + composites)
    # Ordering 4: by number of distinct prime factors (fewer first)
    def _num_distinct_pf(x):
        return len(_prime_factorization(x))
    orderings.append(sorted(candidates, key=lambda x: (_num_distinct_pf(x), x)))
    # Ordering 5: by largest prime factor (larger first)
    def _largest_pf(x):
        pf = _prime_factorization(x)
        return max(pf.keys()) if pf else 0
    orderings.append(sorted(candidates, key=lambda x: (-_largest_pf(x), x)))
    # Ordering 6: by value descending, primes first among same
    orderings.append(sorted(candidates, key=lambda x: (-x, not _is_prime(x))))
    
    best = []
    for order in orderings:
        if time.time() - start > timeout:
            break
        result = _greedy_add(order)
        if len(result) > len(best):
            best = result
    
    # Try some random orderings with remaining time
    import random
    rng = random.Random(42)
    while time.time() - start < timeout - 5.0:
        order = candidates[:]
        rng.shuffle(order)
        result = _greedy_add(order)
        if len(result) > len(best):
            best = result
    
    # Final verification
    if best and _subset_products_distinct(best):
        return sorted(best)
    
    # Fallback
    return [x for x in range(2, 41) if _is_prime(x)]
# EVOLVE-BLOCK-END
