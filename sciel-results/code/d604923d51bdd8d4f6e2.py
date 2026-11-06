import time
import random
from itertools import combinations

def _can_add(family, mask):
    """Check if mask can be added to family without forming a sunflower."""
    len_f = len(family)
    for i in range(len_f):
        a = family[i]
        am = a & mask
        for j in range(i):
            b = family[j]
            bm = b & mask
            ab = a & b
            if am == bm == ab:
                return False
    return True

def _local_search(family, all_sets, deadline):
    """Try swap-based local search to improve family size."""
    family = family[:]
    in_family = set(family)
    best_size = len(family)
    
    # Try removing one element and adding two from outside
    # First, build list of sets not in family
    outside = [m for m in all_sets if m not in in_family]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(outside)
        
        for idx in range(len(family)):
            if time.time() > deadline:
                return family
            removed = family[idx]
            # Try adding pairs from outside
            for oi1 in range(len(outside)):
                if time.time() > deadline:
                    return family
                m1 = outside[oi1]
                for oi2 in range(oi1 + 1, len(outside)):
                    m2 = outside[oi2]
                    # Check if m1 and m2 can both be added with removed gone
                    temp = [x for i, x in enumerate(family) if i != idx]
                    if _can_add(temp, m1) and _can_add(temp + [m1], m2):
                        # This swap increases size by 1
                        family = temp + [m1, m2]
                        in_family = set(family)
                        outside = [m for m in all_sets if m not in in_family]
                        best_size = len(family)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    # Elements are 0..15 corresponding to 1..16
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    best_family = []
    best_size = 0
    
    # Use multiple random orderings
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.shuffle(all_sets)
        family = []
        
        for mask in all_sets:
            if time.time() > deadline:
                break
            
            if _can_add(family, mask):
                family.append(mask)
        
        # Try local search refinement
        if len(family) > best_size and time.time() < deadline - 5:
            refined = _local_search(family, all_sets, deadline - 2)
            if len(refined) > len(family):
                family = refined
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            
            # Early exit if we hit the cap
            if best_size >= 250:
                break
    
    # Convert best_family back to list of sorted lists of integers 1..16
    result = []
    for mask in best_family:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
