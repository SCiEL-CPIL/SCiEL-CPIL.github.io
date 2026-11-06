"""B_3[1] set construction: maximize cardinality with unique unordered triple sums."""
import time
import random
import sys
import numpy as np

N = 4096
MAX_SUM = 3 * N

def _triple_sums_involving(x, S):
    """Compute all triple sums that involve x and elements of S (unordered, with repetition)."""
    sums = set()
    # (x, x, x)
    sums.add(3 * x)
    for y in S:
        sums.add(2 * x + y)
        sums.add(x + 2 * y)
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            sums.add(x + S[i] + S[j])
    return sums

def _check_candidate(x, S, existing_sums):
    """Check if x can be added to S without triple-sum collisions."""
    new_sums = _triple_sums_involving(x, S)
    # Count expected unique sums
    n = len(S)
    expected = 1 + 2 * n + n * (n - 1) // 2
    if len(new_sums) < expected:
        return False, None
    if new_sums & existing_sums:
        return False, None
    return True, new_sums

def _all_triple_sums(S):
    """Compute all triple sums for set S."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.add(S[i] + S[j] + S[k])
    return sums

def _greedy_build(deadline, order=None):
    """Greedy construction: iterate through order, add x if valid."""
    S = []
    existing_sums = set()
    if order is None:
        order = range(1, N + 1)
    for x in order:
        if time.time() > deadline:
            break
        valid, new_sums = _check_candidate(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
    return S

def _greedy_optimized(deadline):
    """Optimized greedy: skip candidates that are likely to collide using modular hints."""
    S = []
    existing_sums = set()
    # Track which residue classes mod small primes are "used" to skip likely bad candidates
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        # Quick rejection: if 3*x is already in existing_sums, skip
        if 3 * x in existing_sums:
            continue
        valid, new_sums = _check_candidate(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
    return S

def _greedy_with_backtracking(deadline, seed=0):
    """Greedy with occasional backtracking: if stuck, remove last few and try alternatives."""
    rng = random.Random(seed)
    S = []
    existing_sums = set()
    max_size = 0
    stuck_count = 0
    skip_set = set()
    
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if x in skip_set:
            continue
        valid, new_sums = _check_candidate(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
            stuck_count = 0
            if len(S) > max_size:
                max_size = len(S)
        else:
            stuck_count += 1
            # After being stuck for a while, try backtracking
            if stuck_count > 3 and len(S) > 1:
                # Remove last element and skip it
                removed = S.pop()
                skip_set.add(removed)
                # Recompute existing_sums
                existing_sums = _all_triple_sums(S)
                stuck_count = 0
    return S

def _local_search(S, existing_sums, deadline, seed=0):
    """Local search: try swaps to increase size or maintain size with smaller elements."""
    rng = random.Random(seed)
    S = S[:]
    existing_sums = set(existing_sums)
    best_size = len(S)
    
    # Phase 1: try to add elements
    added_any = True
    while added_any and time.time() < deadline:
        added_any = False
        for x in range(1, N + 1):
            if x in S:
                continue
            valid, new_sums = _check_candidate(x, S, existing_sums)
            if valid:
                S.append(x)
                S.sort()
                existing_sums |= new_sums
                best_size = len(S)
                added_any = True
                break
    
    # Phase 2: try swaps - remove one element, try to add two
    for _ in range(50):
        if time.time() > deadline:
            break
        if len(S) < 2:
            break
        
        idx = rng.randint(0, len(S) - 1)
        removed = S[idx]
        S.pop(idx)
        existing_sums = _all_triple_sums(S)
        
        # Try to add two elements
        added = 0
        for x in range(1, N + 1):
            if x == removed or x in S:
                continue
            valid, new_sums = _check_candidate(x, S, existing_sums)
            if valid:
                S.append(x)
                existing_sums |= new_sums
                added += 1
                if added == 2:
                    break
            if time.time() > deadline:
                break
        
        S.sort()
        if len(S) > best_size:
            best_size = len(S)
        elif len(S) <= best_size - 1:
            # Revert
            S.append(removed)
            S.sort()
            existing_sums = _all_triple_sums(S)
    
    # Final: try to add more
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if x in S:
            continue
        valid, new_sums = _check_candidate(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
    S.sort()
    return S

def _power_greedy(deadline):
    """Greedy using power-of-3 style spacing for initial candidates, then fill gaps."""
    S = []
    existing_sums = set()
    
    # Phase 1: power-of-3 style
    x = 1
    gap = 1
    while x <= N:
        if time.time() > deadline:
            break
        valid, new_sums = _check_candidate(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
        gap += 1
        x += gap
    
    # Phase 2: fill in with increasing order
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if x in S:
            continue
        valid, new_sums = _check_candidate(x, S, existing_sums)
        if valid:
            S.append(x)
            existing_sums |= new_sums
    S.sort()
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best = []
    
    # Strategy 1: Standard greedy increasing
    S1 = _greedy_build(deadline)
    if len(S1) > len(best):
        best = S1[:]
    
    # Strategy 2: Optimized greedy
    if time.time() < deadline:
        S2 = _greedy_optimized(deadline)
        if len(S2) > len(best):
            best = S2[:]
    
    # Strategy 3: Power greedy
    if time.time() < deadline:
        S3 = _power_greedy(deadline)
        if len(S3) > len(best):
            best = S3[:]
    
    # Strategy 4: Greedy with backtracking
    if time.time() < deadline:
        S4 = _greedy_with_backtracking(deadline, seed=42)
        if len(S4) > len(best):
            best = S4[:]
    
    # Strategy 5: Local search from best
    if time.time() < deadline:
        S5 = _local_search(best, _all_triple_sums(best), deadline, seed=123)
        if len(S5) > len(best):
            best = S5[:]
    
    # Strategy 6: Another local search with different seed
    if time.time() < deadline:
        S6 = _local_search(best, _all_triple_sums(best), deadline, seed=999)
        if len(S6) > len(best):
            best = S6[:]
    
    return best
