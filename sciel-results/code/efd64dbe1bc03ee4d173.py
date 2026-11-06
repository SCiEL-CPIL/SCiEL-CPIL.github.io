"""Improved program: use strategic ordering and greedy with lookahead."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a // gcd(a, b) * b

def _conflict_count(x, selected, lcm_cache):
    """Count how many pairs in selected would form an equal-LCM triple with x."""
    conflicts = 0
    for a, b in combinations(selected, 2):
        lab = lcm_cache.get((a, b), _lcm(a, b))
        lax = lcm_cache.get((a, x), _lcm(a, x))
        lbx = lcm_cache.get((b, x), _lcm(b, x))
        if lab == lax == lbx:
            conflicts += 1
    return conflicts

def _can_add(x, selected, lcm_cache):
    """Check if x can be added without creating an equal-LCM triple."""
    for a, b in combinations(selected, 2):
        lab = lcm_cache.get((a, b), _lcm(a, b))
        lax = lcm_cache.get((a, x), _lcm(a, x))
        lbx = lcm_cache.get((b, x), _lcm(b, x))
        if lab == lax == lbx:
            return False
    return True

def _num_omega(n):
    """Count distinct prime factors."""
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    # Precompute LCM cache for all pairs
    lcm_cache = {}
    for a in range(1, 129):
        for b in range(a + 1, 129):
            lcm_cache[(a, b)] = _lcm(a, b)
            lcm_cache[(b, a)] = lcm_cache[(a, b)]
    
    # Strategy: order elements by number of prime factors (fewer first tend to be less conflicting)
    # Then use greedy with the ability to skip and try later
    elements = list(range(1, 129))
    
    # Sort by number of distinct prime factors, then by value
    elements.sort(key=lambda x: (_num_omega(x), x))
    
    selected = []
    for x in elements:
        if time.time() > deadline:
            break
        if _can_add(x, selected, lcm_cache):
            selected.append(x)
    
    # Local search: try to swap out elements to fit more in
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            current_size = len(selected)
            for i in range(len(selected)):
                if time.time() > deadline - 2:
                    break
                # Try removing selected[i] and see if we can add more
                candidate = selected[:i] + selected[i+1:]
                for x in elements:
                    if x in candidate:
                        continue
                    if _can_add(x, candidate, lcm_cache):
                        new_candidate = candidate + [x]
                        if len(new_candidate) > current_size:
                            selected = new_candidate
                            current_size = len(selected)
                            improved = True
                            break
    
    return selected
# EVOLVE-BLOCK-END
