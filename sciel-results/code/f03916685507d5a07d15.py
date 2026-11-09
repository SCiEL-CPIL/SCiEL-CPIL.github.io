"""Program using a model-and-optimize approach for large integer sets with unique triple sums."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def _initial_sums(initial_set):
    """Compute all triple sums for an initial set."""
    sums = set()
    n = len(initial_set)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = initial_set[i] + initial_set[j] + initial_set[k]
                sums.add(s)
    return sums

def _greedy_incremental(start_list, limit, time_limit):
    """
    Greedy construction using incremental sum-set checks.
    Much faster than full re-verification.
    """
    base = sorted(start_list)
    if not base:
        return []
    
    # Initial check
    all_sums = _initial_sums(base)
    
    # If initial set is invalid, return empty or try to fix? 
    # For this function, we assume input is valid or we check it.
    # Let's add a quick check for validity of initial set if it's small.
    # But _initial_sums doesn't check for duplicates, it just adds to a set.
    # We need to ensure the initial set itself is a B3 set.
    # We'll assume the caller provides a valid start or we check it.
    # To be safe, let's do a quick check for the start list if len < 10.
    if len(base) < 10:
        # Full check for small sets
        seen = set()
        valid = True
        for a in range(len(base)):
            for b in range(a, len(base)):
                for c in range(b, len(base)):
                    s = base[a] + base[b] + base[c]
                    if s in seen:
                        valid = False
                        break
                    seen.add(s)
            if not valid: break
        if not valid:
            return []
        # Re-compute sums correctly if it passed
        all_sums = _initial_sums(base)

    while len(base) < 80:
        next_min = max(base) + 1
        found = False
        
        # Optimization: Only check candidates up to a certain range or until found?
        # Usually the next valid number is close to the max.
        # We scan upwards.
        for x in range(next_min, limit + 1):
            if time.time() > time_limit:
                return base
            
            # Generate new triple sums involving x
            # Triples are (x, x, x), (x, x, a), (x, a, b) with a, b in base, a <= b
            new_sums = set()
            conflict = False
            
            # 1. (x, x, x)
            s_0 = 3 * x
            if s_0 in all_sums or s_0 in new_sums:
                conflict = True
            else:
                new_sums.add(s_0)
            
            if not conflict:
                # 2. (x, x, a) for a in base
                for a in base:
                    s_1 = 2 * x + a
                    if s_1 in all_sums or s_1 in new_sums:
                        conflict = True
                        break
                    new_sums.add(s_1)
            
            if not conflict:
                # 3. (x, a, b) for a, b in base, a <= b
                # Note: We must also ensure (x, a, b) doesn't collide with (x, c, d)
                # which is handled by `s in new_sums` check.
                n = len(base)
                for i in range(n):
                    ai = base[i]
                    for j in range(i, n):
                        bj = base[j]
                        s_2 = x + ai + bj
                        if s_2 in all_sums or s_2 in new_sums:
                            conflict = True
                            break
                        new_sums.add(s_2)
                    if conflict:
                        break
            
            if not conflict:
                # Valid candidate
                base.append(x)
                all_sums.update(new_sums)
                found = True
                break
        
        if not found:
            break
            
    return base

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin out of 60s
    
    limit = 4096
    max_size = 80
    
    best_result = []
    
    # Phase 1: Standard greedy from 1
    # Starting with [1] is a good heuristic for B-h sets.
    result = _greedy_incremental([1], limit, deadline)
    if len(result) > len(best_result):
        best_result = result
        
    if time.time() > deadline:
        return best_result

    # Phase 2: Try other starting points if time permits
    # Sometimes starting with slightly different small sets can lead to larger final sets
    # if the greedy path from 1 gets stuck in a local minimum (though less likely with incremental check).
    
    # We can try starting with [1, 2], [1, 3], etc.
    # But [1] usually dominates. Let's try a few random restarts with small valid seeds.
    
    # Generate a few small valid B3 sets to use as seeds
    seeds_to_try = [[1], [1, 2], [1, 3], [1, 4], [2, 3, 5]]
    
    for seed_set in seeds_to_try:
        if time.time() > deadline:
            break
        
        # Check if seed is valid
        if len(seed_set) > 1:
            # Quick validity check
            sums = set()
            valid = True
            n = len(seed_set)
            for i in range(n):
                for j in range(i, n):
                    for k in range(j, n):
                        s = seed_set[i] + seed_set[j] + seed_set[k]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                if not valid: break
            if not valid:
                continue
                
        result = _greedy_incremental(seed_set, limit, deadline)
        if len(result) > len(best_result):
            best_result = result
            
    return best_result

# EVOLVE-BLOCK-END
