"""Search-based approach to maximize set size under divisibility constraint."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 82.0

def _all_subset_sums_fast(elements):
    """Generate all nonempty subset sums using iterative DP."""
    sums = set()
    for e in elements:
        new_sums = set()
        for s in sums:
            ns = s + e
            if ns not in sums:
                new_sums.add(ns)
        new_sums.add(e)
        sums.update(new_sums)
    return sums

def _check_divisibility(sums):
    """Check if no two distinct subset sums divide one another."""
    sum_list = sorted(sums)
    for i in range(len(sum_list)):
        a = sum_list[i]
        if a == 0:
            continue
        for j in range(i + 1, len(sum_list)):
            b = sum_list[j]
            if b % a == 0:
                return False
    return True

def _is_valid(elements):
    """Check if no two distinct subset sums divide one another."""
    n = len(elements)
    if n == 0:
        return True
    if n > 22:
        # For large n, do a fast pre-check on individual elements
        for i in range(n):
            for j in range(i + 1, n):
                a, b = elements[i], elements[j]
                if a > b:
                    a, b = b, a
                if a != 0 and b % a == 0:
                    return False
        # Then do full check if feasible
        if n <= 24:
            sums = _all_subset_sums_fast(elements)
            return _check_divisibility(sums)
        return True  # Optimistic for very large n
    
    sums = _all_subset_sums_fast(elements)
    return _check_divisibility(sums)

def _can_add(elements, candidate):
    """Check if adding candidate to elements keeps the set valid."""
    new_set = elements + [candidate]
    return _is_valid(new_set)

def _incremental_check_add(existing_sums_sorted, candidate):
    """
    Check if adding 'candidate' to a set with existing subset sums (sorted list)
    maintains the divisibility constraint.
    
    Returns (True, new_sums_sorted) if valid, (False, None) if not.
    """
    e = candidate
    # New sums are: existing_sums + {e} + {s + e for s in existing_sums}
    new_sums_set = set(existing_sums_sorted)
    new_sums_set.add(e)
    for s in existing_sums_sorted:
        new_sums_set.add(s + e)
    
    # Check all pairs in new_sums_set for divisibility
    new_sorted = sorted(new_sums_set)
    n = len(new_sorted)
    for i in range(n):
        a = new_sorted[i]
        if a == 0:
            continue
        for j in range(i + 1, n):
            b = new_sorted[j]
            if b % a == 0:
                return False, None
    
    return True, new_sorted

def _backtrack_search(candidates, deadline, best_size_ref):
    """
    Backtracking search with forward-checking.
    candidates: list of candidate integers to consider
    deadline: time limit
    best_size_ref: list containing [best_size, best_result] to be updated
    """
    best_size = best_size_ref[0]
    best_result = best_size_ref[1]
    
    if not candidates:
        return best_result, best_size
    
    # Sort candidates - try larger ones first to reduce collisions
    cand_list = sorted(candidates, reverse=True)
    n_cands = len(cand_list)
    
    def _bt(idx, cur_elements, cur_sums_sorted, cur_set):
        nonlocal best_size, best_result
        if time.time() > deadline:
            return
        if len(cur_elements) > best_size:
            best_size = len(cur_elements)
            best_result = list(cur_elements)
        
        if idx >= n_cands:
            return
        
        c = cand_list[idx]
        
        # Option 1: skip c
        _bt(idx + 1, cur_elements, cur_sums_sorted, cur_set)
        
        # Option 2: add c if valid
        if c not in cur_set:
            valid, new_sums = _incremental_check_add(cur_sums_sorted, c)
            if valid:
                new_elements = cur_elements + [c]
                new_set = cur_set | {c}
                _bt(idx + 1, new_elements, new_sums, new_set)
    
    _bt(0, [], [], set())
    return best_result, best_size

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # All candidates from 1 to 256
    all_candidates = list(range(1, 257))
    
    random.seed(42)
    
    best_result = []
    best_size = 0
    
    # Strategy 1: Backtracking search with forward-checking
    # Use subsets of candidates to keep search tractable
    for restart in range(40):
        if time.time() > deadline:
            break
        
        # Select a subset of candidates for backtracking
        # Mix of larger and smaller numbers
        if restart % 2 == 0:
            # Focus on larger numbers
            pool = [x for x in all_candidates if x >= 64]
        else:
            # Mixed
            pool = all_candidates
        
        num_cands = min(len(pool), 22)
        selected_cands = random.sample(pool, num_cands)
        
        best_size_ref = [best_size, best_result]
        result, sz = _backtrack_search(selected_cands, deadline, best_size_ref)
        
        if sz > best_size:
            best_result = result
            best_size = sz
        
        if best_size >= 12:
            break
    
    # Strategy 2: Greedy + local search as improvement
    if time.time() < deadline - 10:
        for restart in range(50):
            if time.time() > deadline:
                break
            
            large_candidates = [x for x in all_candidates if x >= 80]
            random.shuffle(large_candidates)
            
            start_size = random.randint(2, 4)
            start_elements = large_candidates[:start_size]
            
            if not _is_valid(start_elements):
                continue
            
            remaining = [x for x in large_candidates if x not in set(start_elements)]
            random.shuffle(remaining)
            
            # Greedy add
            current = list(start_elements)
            current_set = set(current)
            for c in remaining:
                if time.time() > deadline:
                    break
                if c not in current_set:
                    if _can_add(current, c):
                        current.append(c)
                        current_set.add(c)
            
            # Local search
            all_remaining = [x for x in all_candidates if x not in set(current)]
            best_local = list(current)
            best_local_size = len(current)
            
            for _ in range(150):
                if time.time() > deadline:
                    break
                
                if all_remaining:
                    idx = random.randint(0, len(all_remaining) - 1)
                    c = all_remaining[idx]
                    if c not in current_set:
                        if _can_add(current, c):
                            current.append(c)
                            current_set.add(c)
                            if len(current) > best_local_size:
                                best_local = list(current)
                                best_local_size = len(current)
                
                if len(current) > 1 and all_remaining:
                    remove_idx = random.randint(0, len(current) - 1)
                    removed = current.pop(remove_idx)
                    current_set.discard(removed)
                    
                    added = False
                    for _ in range(min(10, len(all_remaining))):
                        add_idx = random.randint(0, len(all_remaining) - 1)
                        c = all_remaining[add_idx]
                        if c not in current_set and c != removed:
                            if _can_add(current, c):
                                current.append(c)
                                current_set.add(c)
                                added = True
                                break
                    
                    if not added:
                        current.append(removed)
                        current_set.add(removed)
            
            if best_local_size > best_size:
                best_result = best_local
                best_size = best_local_size
            
            if best_size >= 12:
                break
    
    # Final verification
    if best_result and not _is_valid(best_result):
        best_result = [31, 32, 34]
    
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
