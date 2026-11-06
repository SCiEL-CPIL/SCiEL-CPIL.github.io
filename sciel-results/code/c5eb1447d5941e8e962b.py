"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _check_addition(values, sums, x):
    """Check if x can be appended to values without sum collision.
    Returns (ok, new_sums_set) if ok, else (False, None)."""
    tails = [x]
    total = x
    for a in reversed(values):
        total += a
        tails.append(total)
    for t in tails:
        if t in sums:
            return False, None
    return True, set(tails)

def _greedy(values, sums, start, end):
    """Greedy: try values from start to end, append if possible."""
    for x in range(start, end + 1):
        ok, new_sums = _check_addition(values, sums, x)
        if ok:
            values.append(x)
            sums.update(new_sums)
    return values

def _greedy_from(values, sums, start, end):
    """Greedy starting from a given partial sequence."""
    return _greedy(values[:], sums.copy(), start, end)

def _dfs_extend(values, sums, candidates, max_depth, deadline):
    """DFS to extend sequence, trying candidates in order."""
    best = list(values)
    best_sums = set(sums)
    
    def dfs(vals, sm, cand, depth):
        nonlocal best, best_sums
        if time.time() > deadline:
            return
        if len(vals) > len(best):
            best = list(vals)
            best_sums = set(sm)
        if depth >= max_depth or not cand:
            return
        for i, x in enumerate(cand):
            if x <= vals[-1]:
                continue
            ok, new_sums = _check_addition(vals, sm, x)
            if ok:
                new_vals = vals + [x]
                new_sm = sm | new_sums
                dfs(new_vals, new_sm, cand[i+1:], depth + 1)
                if time.time() > deadline:
                    return
    
    dfs(values, sums, candidates, 0, deadline)
    return best

def construct():
    deadline = time.time() + 50.0  # 50s budget, 10s safety margin
    best_seq = []
    best_len = 0
    
    # Strategy 1: Standard greedy (smallest first)
    values = []
    sums = set()
    for x in range(1, 257):
        ok, new_sums = _check_addition(values, sums, x)
        if ok:
            values.append(x)
            sums.update(new_sums)
    if len(values) > best_len:
        best_seq = values[:]
        best_len = len(values)
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 2: Greedy with different starting offsets
    for start_offset in range(0, 20):
        if time.time() > deadline:
            break
        values = []
        sums = set()
        for x in range(1 + start_offset, 257):
            ok, new_sums = _check_addition(values, sums, x)
            if ok:
                values.append(x)
                sums.update(new_sums)
        if len(values) > best_len:
            best_seq = values[:]
            best_len = len(values)
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 3: DFS extension from greedy results
    # Use the best greedy result and try to extend/modify
    base_values = best_seq[:]
    base_sums = set()
    # Recompute sums for base
    for i in range(len(base_values)):
        s = 0
        for j in range(i, len(base_values)):
            s += base_values[j]
            base_sums.add(s)
    
    # Try DFS with remaining candidates
    remaining = [x for x in range(1, 257) if x not in base_values and x > base_values[-1]]
    if remaining:
        extended = _dfs_extend(base_values, base_sums, remaining, 5, deadline)
        if len(extended) > best_len:
            best_seq = extended
            best_len = len(extended)
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 4: Try greedy with larger gaps (skip some small numbers)
    for skip_pattern in range(0, 10):
        if time.time() > deadline:
            break
        values = []
        sums = set()
        for x in range(1, 257):
            # Skip based on pattern
            if skip_pattern > 0 and x <= skip_pattern:
                continue
            ok, new_sums = _check_addition(values, sums, x)
            if ok:
                values.append(x)
                sums.update(new_sums)
        if len(values) > best_len:
            best_seq = values[:]
            best_len = len(values)
    
    if time.time() > deadline:
        return best_seq
    
    # Strategy 5: More aggressive DFS from scratch with limited depth
    # Start with a few initial values and DFS
    for first in [1, 2, 3, 5, 8, 13]:
        if time.time() > deadline:
            break
        values = [first]
        sums = {first}
        candidates = list(range(first + 1, 257))
        extended = _dfs_extend(values, sums, candidates, 8, deadline)
        if len(extended) > best_len:
            best_seq = extended
            best_len = len(extended)
    
    return best_seq
# EVOLVE-BLOCK-END
