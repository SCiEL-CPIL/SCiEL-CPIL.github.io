"""Search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import sys
import random
from math import gcd
from functools import reduce

# LCM of 1..22
def _lcm(a, b):
    return a * b // gcd(a, b)

_LCM = reduce(_lcm, range(1, 23))
# LCM = 232792560

# Weights: w_i = LCM // i for i in 1..22
_WEIGHTS = [None] + [_LCM // i for i in range(1, 23)]

def _can_add(sums, w):
    """Check if weight w can be added to the set of sums without collision."""
    for s in sums:
        if (s + w) in sums:
            return False
    return True

def _add_weight(sums, w):
    """Add weight w to the set of sums, returning the new set."""
    return sums | {s + w for s in sums}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_subset)
    
    indices = list(range(1, 23))
    rng = random.Random(42)
    
    # Phase 1: Randomized Greedy Construction
    # We try to build a large set from scratch using random orderings.
    
    # Initial deterministic attempts
    orderings = [
        indices,
        sorted(indices, reverse=True),
        sorted(indices, key=lambda x: _WEIGHTS[x]),
        sorted(indices, key=lambda x: -_WEIGHTS[x]),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        current = []
        sums = {0}
        for i in order:
            w = _WEIGHTS[i]
            if _can_add(sums, w):
                sums = _add_weight(sums, w)
                current.append(i)
        
        if len(current) > best_size:
            best_size = len(current)
            best_subset = list(current)
            
    # Phase 2: Extensive Randomized Greedy
    # Each restart: Shuffle indices, greedily add.
    # Cost per restart: ~22 * (2^k) checks.
    # If k=16, 22 * 65536 ~ 1.4M ops.
    # Python can do ~10-50M simple ops/sec.
    # So ~0.1s per restart.
    # We have ~80s. So ~800 restarts.
    
    restarts = 0
    while time.time() < deadline - 5:
        perm = indices[:]
        rng.shuffle(perm)
        
        current = []
        sums = {0}
        for i in perm:
            w = _WEIGHTS[i]
            if _can_add(sums, w):
                sums = _add_weight(sums, w)
                current.append(i)
        
        if len(current) > best_size:
            best_size = len(current)
            best_subset = list(current)
            # If we find a very large set, we might want to stop?
            # But we want max. 22 is the max.
            # If we hit 20, we are likely near the end.
        
        restarts += 1
        
        # If we've done enough restarts, break to allow local search
        if restarts > 500 and time.time() > deadline - 10:
             break

    # Phase 3: Local Search / Growth on the Best Subset Found
    # Try to add more elements to best_subset.
    # We do this by trying to add elements from the complement in random orders.
    
    best_set_set = set(best_subset)
    
    # We perform multiple "growth" attempts
    growth_attempts = 0
    while time.time() < deadline:
        curr = list(best_subset)
        curr_set = set(curr)
        sums = {0}
        for i in curr:
            sums = _add_weight(sums, _WEIGHTS[i])
            
        avail = [i for i in range(1, 23) if i not in curr_set]
        if not avail:
            break
            
        rng.shuffle(avail)
        
        # Greedily add from avail
        for i in avail:
            w = _WEIGHTS[i]
            if _can_add(sums, w):
                sums = _add_weight(sums, w)
                curr.append(i)
                curr_set.add(i)
                
        if len(curr) > best_size:
            best_size = len(curr)
            best_subset = list(curr)
            best_set_set = set(best_subset)
            
        growth_attempts += 1
        if growth_attempts > 200:
            break

    return best_subset
