"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def _is_valid_with(x, selected):
    """
    Check if x can be added to selected without violating constraints.
    selected is a list of integers.
    Returns True if valid.
    """
    n = len(selected)
    if n < 3:
        return True
    
    # Check 1: For any a in selected, a does not divide x + y + z
    # where y, z are distinct elements in selected \ {a}
    # Note: y, z must be distinct from each other and from a.
    # Since elements are distinct in selected, we just need y != a and z != a.
    
    # Optimization: If x is large, sums x+y+z are large.
    # If a is small, it's more likely to divide.
    
    # Iterate a
    for a in selected:
        # We need to check if there exist y, z in selected \ {a} such that (x + y + z) % a == 0
        # This is equivalent to (x + y + z) % a == 0
        # Let rem = x % a. We need (y + z) % a == (-rem) % a.
        
        target = (-x) % a
        # Check pairs from selected excluding a
        # To avoid O(N^2) per a, we can just iterate pairs.
        # N is small (<=48).
        
        # Find indices of a? No, just check value.
        # Since values are distinct, we can filter.
        
        # Efficient pair check:
        # Create list of others
        others = [v for v in selected if v != a]
        m = len(others)
        if m < 2:
            continue
            
        # Check all pairs in others
        # If m is ~47, m*(m-1)/2 ~ 1000 pairs.
        # 48 * 1000 = 48,000 ops per x.
        # 192 x's = 9M ops.
        
        found = False
        for i in range(m):
            y = others[i]
            # We can break early if we find a match? No, we need to know if ANY match exists to invalidate.
            # So if we find one, we return False.
            for j in range(i+1, m):
                z = others[j]
                if (x + y + z) % a == 0:
                    return False
        # If no violation for this a, continue to next a
        
    # Check 2: x does not divide a + y + z for any distinct a, y, z in selected
    # Iterate triples
    # Number of triples: 48 * 47 * 46 / 6 ~ 18,000
    # 192 * 18,000 = 3.4M ops.
    
    for a, y, z in combinations(selected, 3):
        if (a + y + z) % x == 0:
            return False
            
    return True

def _greedy(ordering):
    selected = []
    remaining = set(ordering)
    
    while remaining and len(selected) < 48:
        # Pick the first valid one from the ordered list
        added = False
        for x in ordering:
            if x in remaining:
                if _is_valid_with(x, selected):
                    selected.append(x)
                    remaining.remove(x)
                    added = True
                    break
        if not added:
            break
            
    return selected

def _local_search(initial, time_limit=50.0):
    start = time.time()
    selected = list(initial)
    selected_set = set(selected)
    all_nums = set(range(1, 193))
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # 1. Try to add
        candidates = [x for x in all_nums if x not in selected_set]
        random.shuffle(candidates)
        
        added = False
        for x in candidates:
            if _is_valid_with(x, selected):
                selected.append(x)
                selected_set.add(x)
                added = True
                improved = True
                break
        
        if added:
            continue
            
        # 2. Try to swap
        # Remove one, add one
        # To make this efficient, we can try a few random swaps
        # Or iterate through removals
        
        swapped = False
        # Shuffle selected to randomize which one we remove
        rem_candidates = list(selected)
        random.shuffle(rem_candidates)
        
        for u in rem_candidates:
            if time.time() - start > time_limit:
                break
                
            # Temporarily remove u
            selected_temp = [x for x in selected if x != u]
            
            # Try to add any valid x to selected_temp
            # We can reuse the candidate list logic
            # But we need to check against selected_temp
            
            add_found = False
            # Use a subset of candidates to save time?
            # Let's try all
            for x in candidates:
                if _is_valid_with(x, selected_temp):
                    # Perform swap
                    selected = selected_temp + [x]
                    selected_set.remove(u)
                    selected_set.add(x)
                    swapped = True
                    add_found = True
                    improved = True
                    break
                    
            if add_found:
                break
                
        if not swapped:
            # If we couldn't add or swap, try a more aggressive search?
            # Or just break
            break
            
    return selected

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin
    
    best_set = []
    
    # Strategy 1: Largest First Greedy
    if time.time() < deadline - 10:
        s = _greedy(range(192, 0, -1))
        if len(s) > len(best_set):
            best_set = s
            
    # Strategy 2: Smallest First Greedy
    if time.time() < deadline - 10:
        s = _greedy(range(1, 193))
        if len(s) > len(best_set):
            best_set = s
            
    # Strategy 3: Random Greedy + Local Search
    # Run local search on the best greedy result
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        best_set = _local_search(best_set, time_limit=remaining_time)
        
    # If time remains, try random restarts
    if time.time() < deadline - 5:
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            random.seed(seed)
            ordering = list(range(1, 193))
            random.shuffle(ordering)
            s = _greedy(ordering)
            if len(s) > len(best_set):
                best_set = s
            elif len(s) == len(best_set) and time.time() < deadline - 10:
                # Try local search on this too if it's same size
                rem = deadline - 10 - time.time()
                if rem > 2:
                    s_ls = _local_search(s, time_limit=rem)
                    if len(s_ls) > len(best_set):
                        best_set = s_ls
                        
    return best_set
# EVOLVE-BLOCK-END
