"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 40, 'maximum_size': 18}

def construct():
    """
    Find the largest subset of {2, ..., 40} such that all 2^k subset products are distinct.
    Uses fast integer-product collision checking with diverse random restarts.
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin within 60s limit
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Pre-compute complexity metrics
    def complexity(num):
        cnt = 0
        d = 2
        m = num
        while d * d <= m:
            while m % d == 0:
                cnt += 1
                m //= d
            d += 1
        if m > 1:
            cnt += 1
        return cnt

    complexities = {x: complexity(x) for x in candidates}
    
    # Pre-compute distinct prime factor count
    def distinct_pf(num):
        cnt = 0
        d = 2
        m = num
        while d * d <= m:
            if m % d == 0:
                cnt += 1
                while m % d == 0:
                    m //= d
            d += 1
        if m > 1:
            cnt += 1
        return cnt

    dpf = {x: distinct_pf(x) for x in candidates}
    
    best_set = []
    best_len = 0
    
    # Pre-compute several deterministic orderings
    order_by_complexity = sorted(candidates, key=lambda x: (complexities[x], x))
    order_asc = sorted(candidates)
    order_desc = sorted(candidates, reverse=True)
    order_primes_first = sorted(candidates, key=lambda x: (dpf[x], complexities[x], x))
    order_by_dpf_desc = sorted(candidates, key=lambda x: (-dpf[x], x))
    
    num_restarts = 500000  # High number, but loop breaks on time
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Choose a strategy for ordering
        r = random.random()
        if r < 0.12:
            ordered = order_by_complexity
        elif r < 0.24:
            ordered = order_asc
        elif r < 0.36:
            ordered = order_desc
        elif r < 0.48:
            ordered = order_primes_first
        elif r < 0.60:
            ordered = order_by_dpf_desc
        elif r < 0.75:
            ordered = candidates[:]
            random.shuffle(ordered)
        elif r < 0.87:
            # Partially random: sort by random key
            keys = {x: random.random() for x in candidates}
            ordered = sorted(candidates, key=lambda x: keys[x])
        else:
            # Reverse of a random permutation
            ordered = candidates[:]
            random.shuffle(ordered)
            ordered.reverse()
            
        current_set = []
        products = {1}
        
        for cand in ordered:
            # Check if adding cand keeps all products distinct
            # Collision iff exists p1, p2 in products such that p1 * cand == p2
            valid = True
            for p in products:
                if p * cand in products:
                    valid = False
                    break
            
            if valid:
                current_set.append(cand)
                new_prods = [p * cand for p in products]
                products.update(new_prods)
                
                if len(current_set) > best_len:
                    best_len = len(current_set)
                    best_set = current_set[:]
                    
                    if best_len >= 18:
                        # Found 18, which is likely optimal. Keep searching briefly for 19.
                        pass
    
    # Fallback if no set found (should not happen)
    if not best_set:
        best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Final validation using actual big integer products to be absolutely sure
    if best_set:
        prods = {1}
        valid = True
        for x in best_set:
            new_prods = [p * x for p in prods]
            for np in new_prods:
                if np in prods:
                    valid = False
                    break
            if not valid:
                break
            prods.update(new_prods)
        
        if not valid:
            # This should not happen, but if it does, fall back to a known valid set
            best_set = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    return list(best_set)
# EVOLVE-BLOCK-END
