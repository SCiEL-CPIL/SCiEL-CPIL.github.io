"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 256

def _find_conflicts(s):
    """
    Returns a dictionary mapping each element in s to the number of conflicts it's involved in.
    A conflict is a triple (a, b, c) with a < b <= c in s such that a | (b+c).
    """
    s_set = set(s)
    s_sorted = sorted(s)
    conflict_counts = {x: 0 for x in s}
    
    # For each a in s
    for i, a in enumerate(s_sorted):
        if a == 0:
            continue
        # b must be > a
        # c must be >= b
        # We iterate b and c
        # To speed up, we can break early if b is too large? No.
        # Just iterate all pairs (b, c) with b > a, c >= b
        # This is O(N^3) worst case.
        for j in range(i + 1, len(s_sorted)):
            b = s_sorted[j]
            # c ranges from b to max
            # We need (b+c) % a == 0
            target = (-b) % a
            # Check c in s_sorted[j:]
            for k in range(j, len(s_sorted)):
                c = s_sorted[k]
                if c % a == target:
                    # Conflict found: a, b, c
                    conflict_counts[a] += 1
                    conflict_counts[b] += 1
                    conflict_counts[c] += 1
    return conflict_counts

def _resolve_conflicts(s):
    """
    Repeatedly remove the element with the highest conflict count until valid.
    """
    s = list(s)
    while True:
        conflict_counts = _find_conflicts(s)
        if not any(conflict_counts[x] > 0 for x in s):
            break
        # Find max conflict
        max_elem = max(s, key=lambda x: conflict_counts[x])
        s.remove(max_elem)
    return s

def _try_expand(s):
    """
    Try to add elements from 1 to N not in s, if they don't create conflicts.
    """
    s = list(s)
    s_set = set(s)
    changed = True
    while changed:
        changed = False
        for candidate in range(1, N + 1):
            if candidate in s_set:
                continue
            # Check if adding candidate creates a conflict
            # New set: s + [candidate]
            # We only need to check conflicts involving candidate
            new_s = s + [candidate]
            new_s.sort()
            valid = True
            cand_idx = new_s.index(candidate)
            
            # 1. candidate is 'a'
            # a < b <= c, a divides b+c
            # Here a = candidate. b, c > candidate.
            for j in range(cand_idx + 1, len(new_s)):
                b = new_s[j]
                target = (-b) % candidate
                for k in range(j, len(new_s)):
                    c = new_s[k]
                    if c % candidate == target:
                        valid = False
                        break
                if not valid:
                    break
            
            # 2. candidate is 'b'
            # a < b <= c, a divides b+c
            # Here b = candidate. a < candidate, c >= candidate.
            if valid:
                for i in range(cand_idx):
                    a = new_s[i]
                    if a == 0:
                        continue
                    # b = candidate
                    # c >= candidate
                    # Check c in new_s[cand_idx:]
                    target = (-candidate) % a
                    for k in range(cand_idx, len(new_s)):
                        c = new_s[k]
                        if c % a == target:
                            valid = False
                            break
                    if not valid:
                        break
            
            # 3. candidate is 'c'
            # a < b <= c, a divides b+c
            # Here c = candidate. a < b <= candidate.
            if valid:
                for i in range(len(new_s) - 2):
                    a = new_s[i]
                    if a == 0:
                        continue
                    for j in range(i + 1, len(new_s) - 1):
                        b = new_s[j]
                        if (b + candidate) % a == 0:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                s = new_s
                s_set.add(candidate)
                changed = True
    return s

def _local_search_improved(s, time_limit=10.0):
    """
    Local search: Try swapping elements to potentially allow more additions.
    Or just try to add/remove.
    """
    start_time = time.time()
    best = list(s)
    current = list(s)
    current.sort()
    
    while time.time() - start_time < time_limit:
        # Try to add
        added = False
        for candidate in range(1, N + 1):
            if candidate in current:
                continue
            # Quick check: does candidate conflict with any a, b in current?
            # We can use a fast check
            valid = True
            # Check candidate as a
            for b in current:
                if b <= candidate: continue
                for c in current:
                    if c < b: continue
                    if (b+c) % candidate == 0:
                        valid = False
                        break
                if not valid: break
            if not valid: continue
            
            # Check candidate as b
            for a in current:
                if a >= candidate: continue
                for c in current:
                    if c < candidate: continue
                    if (candidate+c) % a == 0:
                        valid = False
                        break
                if not valid: break
            if not valid: continue
            
            # Check candidate as c
            for a in current:
                for b in current:
                    if b <= a: continue
                    if b > candidate: continue # b <= c=candidate
                    if (b+candidate) % a == 0:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                current.append(candidate)
                current.sort()
                added = True
                if len(current) > len(best):
                    best = list(current)
                break # Restart scan after addition
        
        if not added:
            # Try swapping
            # Pick random element to remove
            if len(current) > 0:
                idx = random.randint(0, len(current) - 1)
                removed = current.pop(idx)
                # Try to add a random element
                candidates = [x for x in range(1, N+1) if x not in current]
                if candidates:
                    add_candidate = random.choice(candidates)
                    new_current = current + [add_candidate]
                    new_current.sort()
                    # Check validity
                    if _is_valid_fast(new_current):
                        current = new_current
                        if len(current) > len(best):
                            best = list(current)
                    else:
                        # Revert
                        current.append(removed)
                        current.sort()
    
    return best

def _is_valid_fast(s_sorted):
    """Faster validity check assuming s_sorted is sorted."""
    n = len(s_sorted)
    for i, a in enumerate(s_sorted):
        if a == 0:
            continue
        for j in range(i + 1, n):
            b = s_sorted[j]
            target = (-b) % a
            for k in range(j, n):
                c = s_sorted[k]
                if c % a == target:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_set = list(range(192, 257))
    
    random.seed(42)
    
    # Strategy 1: Conflict Resolution from Upper Tails
    for start_val in [128, 135, 140, 150, 160, 170, 180]:
        if time.time() - start_time > deadline:
            break
        candidate = list(range(start_val, 257))
        # Resolve conflicts
        resolved = _resolve_conflicts(candidate)
        # Expand
        expanded = _try_expand(resolved)
        if len(expanded) > len(best_set):
            best_set = expanded
    
    if time.time() - start_time > deadline:
        return best_set

    # Strategy 2: Local Search on best found
    improved = _local_search_improved(best_set, time_limit=10.0)
    if len(improved) > len(best_set):
        best_set = improved

    if time.time() - start_time > deadline:
        return best_set

    # Strategy 3: Randomized Restart with Conflict Resolution
    for _ in range(10):
        if time.time() - start_time > deadline:
            break
        # Random subset of [100, 256]
        pool = list(range(100, 257))
        random.shuffle(pool)
        size = random.randint(80, 120)
        candidate = pool[:size]
        resolved = _resolve_conflicts(candidate)
        expanded = _try_expand(resolved)
        if len(expanded) > len(best_set):
            best_set = expanded
            # Try local search on this new best
            if time.time() - start_time < deadline - 10:
                improved = _local_search_improved(best_set, time_limit=5.0)
                if len(improved) > len(best_set):
                    best_set = improved

    return best_set
# EVOLVE-BLOCK-END
