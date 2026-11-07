import time
import random
import itertools

_P0 = time.time()
_DEADLINE = _P0 + 85.0

def _subset_products_set(nums):
    prods = {1}
    for x in nums:
        add = set()
        for p in prods:
            add.add(p * x)
        prods = prods | add
    return prods

def _can_add(prods, x):
    # Check if adding x causes a collision
    # New products are p * x for p in prods
    # Collision if any p * x is already in prods
    # Note: p * x == q * y is not the issue, p * x == p' is the issue
    # Wait, the set of products is closed under multiplication by x?
    # If we have set P of products, and we add x, new products are {p*x | p in P}.
    # We need P and {p*x | p in P} to be disjoint.
    for p in prods:
        if (p * x) in prods:
            return False
    return True

def _is_valid_set(nums):
    prods = {1}
    for x in nums:
        if not _can_add(prods, x):
            return False
        new_prods = set(prods)
        for p in prods:
            new_prods.add(p * x)
        prods = new_prods
    return True

def construct():
    # Baseline 16
    baseline = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Try to find 17
    # Random search: Start with baseline, try to swap elements
    rng = random.Random(42)
    all_nums = list(range(2, 41))
    
    # Try adding any remaining number to baseline
    for x in all_nums:
        if x in baseline:
            continue
        prods = _subset_products_set(baseline)
        if _can_add(prods, x):
            return baseline + [x]
            
    # Try swapping one baseline element with two others
    # This is expensive, so do it with a limit
    attempts = 0
    while attempts < 1000 and time.time() < _DEADLINE:
        attempts += 1
        # Pick a random baseline element to remove
        idx = rng.randint(0, 15)
        removed = baseline[idx]
        current = [x for i, x in enumerate(baseline) if i != idx]
        
        # Try to add two numbers from the remaining pool
        pool = [x for x in all_nums if x not in baseline and x != removed]
        # Shuffle pool
        rng.shuffle(pool)
        
        found = False
        for i in range(len(pool)):
            if time.time() > _DEADLINE:
                return baseline
            c1 = pool[i]
            prods1 = _subset_products_set(current)
            if not _can_add(prods1, c1):
                continue
            # Update prods with c1
            new_prods = set(prods1)
            for p in prods1:
                new_prods.add(p * c1)
            
            for j in range(i+1, len(pool)):
                c2 = pool[j]
                if _can_add(new_prods, c2):
                    final_set = current + [c1, c2]
                    if _is_valid_set(final_set):
                        return final_set
        # If no pair found, try adding just one from pool?
        # We already checked adding one to full baseline.
        # But maybe removing one allows adding one?
        for c in pool:
             if time.time() > _DEADLINE:
                return baseline
             prods1 = _subset_products_set(current)
             if _can_add(prods1, c):
                 final_set = current + [c]
                 if len(final_set) == 16:
                     pass # Same size, not improvement
                 # We want 17, so this path is only useful if we can add 2.
                 # But maybe we can add 1 and then another later?
                 # For simplicity, stick to swap 1 for 2.
        
    return baseline
