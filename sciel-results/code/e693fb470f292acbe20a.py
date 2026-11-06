"""Construct a small inclusion-maximal Sidon set in [1,256].

Strategy:
1. Build a strong baseline Sidon set using greedy ascending.
2. Aggressively reduce cardinality via removal: remove an element if the set
   remains maximal (every missing element still conflicts).
3. Use a deadline guard to ensure we finish within 60s.
"""

import time
import random

N = 256

def _is_sidon(set_list):
    """Check if set_list is a Sidon set. Returns True if all pair sums a+b (a<=b) are distinct."""
    s = set(set_list)
    if len(s) != len(set_list):
        return False
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sm = lst[i] + lst[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _compute_sums(set_list):
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sums.add(lst[i] + lst[j])
    return sums

def _is_maximal(set_list, sums, N_val=N):
    """Check if set_list is inclusion-maximal Sidon in [1, N_val]."""
    s = set(set_list)
    for x in range(1, N_val + 1):
        if x in s:
            continue
        # If 2x is already in sums, then adding x would create duplicate 2x = x+x
        if 2 * x in sums:
            continue
        # Check if x + elem is in sums for any elem in set_list
        conflict = False
        for elem in set_list:
            if (x + elem) in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _greedy_ascending():
    """Greedy: pick smallest possible element."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _local_search_reduce(start_set, deadline, rng):
    """Try to reduce cardinality by removing elements that preserve maximality."""
    current = list(start_set)
    current_sums = _compute_sums(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = current[:idx] + current[idx+1:]
            if not candidate:
                continue
            cand_sums = _compute_sums(candidate)
            if _is_maximal(candidate, cand_sums):
                current = candidate
                current_sums = cand_sums
                improved = True
                break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    rng = random.Random(42)
    
    # Generate multiple starting candidates
    candidates = []
    
    # 1. Greedy ascending
    c1 = _greedy_ascending()
    candidates.append(c1)
    
    # 2. Randomized greedy with different orderings
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        selected = []
        sums = set()
        for x in order:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        candidates.append(selected)
    
    # Ensure all candidates are valid Sidon sets
    valid_candidates = []
    for c in candidates:
        if _is_sidon(c):
            valid_candidates.append(c)
    
    if not valid_candidates:
        return _greedy_ascending()
    
    # Sort by cardinality
    valid_candidates.sort(key=len)
    
    best_set = valid_candidates[0]
    best_size = len(best_set)
    
    # Local search reduction on the best candidates
    for c in valid_candidates[:min(5, len(valid_candidates))]:
        if time.time() > deadline:
            break
        reduced = _local_search_reduce(c, deadline, rng)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
    
    # Final verification
    if not _is_sidon(best_set):
        return _greedy_ascending()
    
    # Ensure maximality
    if not _is_maximal(best_set, _compute_sums(best_set)):
        # Add back necessary elements
        current = list(best_set)
        current_sums = _compute_sums(current)
        current_set = set(current)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in range(1, N+1):
                if time.time() > deadline:
                    break
                if x in current_set:
                    continue
                if 2*x in current_sums:
                    continue
                conflict = False
                for elem in current:
                    if (x+elem) in current_sums:
                        conflict = True
                        break
                if not conflict:
                    current.append(x)
                    current_set.add(x)
                    current_sums.update([x+x] + [x+e for e in current[:-1]])
                    changed = True
                    break
        best_set = current
        if not _is_sidon(best_set):
            return _greedy_ascending()
    
    return best_set
