import time
import random
from itertools import combinations

def _subset_sums_with_counts(elements):
    """Return dict of sum -> count using incremental DP."""
    n = len(elements)
    if n == 0:
        return {}
    sums = [0] * (1 << n)
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        sums[mask] = sums[mask ^ lsb] + elements[i]
    sum_count = {}
    for mask in range(1, 1 << n):
        s = sums[mask]
        sum_count[s] = sum_count.get(s, 0) + 1
    return sum_count

def _check_divisibility_fast(sums_list):
    """Check no two distinct sums divide each other."""
    s = sorted(sums_list)
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            if s[j] % si == 0:
                return False
    return True

def _is_valid_fast(elements):
    """Fast validity check with incremental sums."""
    n = len(elements)
    if n == 0:
        return True
    if len(elements) != len(set(elements)):
        return False
    if any(x < 1 or x > 256 for x in elements):
        return False
    sum_count = _subset_sums_with_counts(elements)
    for s, c in sum_count.items():
        if c > 1:
            return False
    sums_list = list(sum_count.keys())
    if not _check_divisibility_fast(sums_list):
        return False
    return True

def _try_add_incremental(current, sums_dict, candidate, deadline):
    """Try adding candidate using precomputed sums. Returns (new_sums_dict, True) or (None, False)."""
    if time.time() > deadline:
        return None, False
    
    # Optimization: If candidate is in current, it's not distinct, but we check distinctness of elements separately.
    # Here we assume candidate is not in current.
    
    new_sums_dict = {}
    # Copy existing sums
    for s, c in sums_dict.items():
        new_sums_dict[s] = new_sums_dict.get(s, 0) + c
    # Add candidate itself
    new_sums_dict[candidate] = new_sums_dict.get(candidate, 0) + 1
    # Add candidate to all existing sums
    for s, c in sums_dict.items():
        ns = s + candidate
        new_sums_dict[ns] = new_sums_dict.get(ns, 0) + c
        
    # Check for duplicate sums
    for s, c in new_sums_dict.items():
        if c > 1:
            return None, False
            
    # Check divisibility
    sums_list = list(new_sums_dict.keys())
    if not _check_divisibility_fast(sums_list):
        return None, False
        
    return new_sums_dict, True

def _greedy_build(candidates, deadline, max_size=30):
    """Greedy: add candidates in given order if valid, using incremental sums."""
    current = []
    sums_dict = {}
    for c in candidates:
        if time.time() > deadline:
            break
        if len(current) >= max_size:
            break
        new_sums, ok = _try_add_incremental(current, sums_dict, c, deadline)
        if ok:
            current.append(c)
            sums_dict = new_sums
    return current

def _local_search_improve(best_set, deadline):
    """Try to improve by adding elements or swapping."""
    current = list(best_set)
    if not current:
        return current
    sums_dict = _subset_sums_with_counts(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        for c in range(1, 257):
            if time.time() > deadline:
                break
            if c in current:
                continue
            new_sums, ok = _try_add_incremental(current, sums_dict, c, deadline)
            if ok:
                current.append(c)
                sums_dict = new_sums
                improved = True
                break
        # If no addition, try swapping
        if not improved:
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                old_val = current[i]
                for c in range(1, 257):
                    if time.time() > deadline:
                        break
                    if c == old_val or c in current:
                        continue
                    # Try swap: remove old_val, add c
                    # This is complex to do incrementally. 
                    # For now, just try adding. Swapping is harder.
                    # We can implement a simple swap by reconstructing sums_dict
                    # but that is expensive. 
                    # Let's stick to adding for now, or simple 1-for-1 swap if we can.
                    pass
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Greedy with large numbers first
    cands_desc = list(range(256, 0, -1))
    result = _greedy_build(cands_desc, deadline)
    if len(result) > best_len:
        best_set = result
        best_len = len(best_set)
        
    # Strategy 2: Randomized greedy
    for attempt in range(500):
        if time.time() > deadline:
            break
        cands = list(range(1, 257))
        random.shuffle(cands)
        result = _greedy_build(cands, deadline)
        if len(result) > best_len:
            best_set = result
            best_len = len(best_set)
            if best_len >= 15: # High bar
                break
                
    # Strategy 3: Local search
    if time.time() < deadline:
        improved_set = _local_search_improve(best_set, deadline)
        if len(improved_set) > best_len:
            best_set = improved_set
            best_len = len(best_set)
            
    return [int(x) for x in best_set]
