"""Rewritten program: fast incremental greedy with multiple orderings and timed local search."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 85.0
_START = None

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _check_no_divisibility(sums):
    """Check that no two distinct values in sums divide one another."""
    sl = sorted(sums)
    n = len(sl)
    for i in range(n):
        ai = sl[i]
        for j in range(i + 1, n):
            if sl[j] % ai == 0:
                return False
    return True

def _subset_sums_from_list(elements):
    """Compute all non-empty subset sums from a list."""
    n = len(elements)
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += elements[i]
        sums.add(s)
    return sums

def _greedy_construct(candidates, deadline_check=None):
    """Greedy: try adding candidates in given order, keeping only valid additions."""
    chosen = []
    current_sums = set()
    
    for x in candidates:
        if deadline_check and deadline_check() < 1.0:
            break
        if x in chosen:
            continue
        # Incrementally compute new sums
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
        
        if _check_no_divisibility(new_sums):
            chosen.append(x)
            current_sums = new_sums
    
    return chosen

def _validate_full(elements):
    """Full validation."""
    if not elements:
        return True
    if len(elements) != len(set(elements)):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    sums = _subset_sums_from_list(elements)
    return _check_no_divisibility(sums)

def construct():
    global _START
    _START = time.time()
    
    pool = list(range(1, 257))
    best = []
    best_len = 0
    
    # Multiple greedy orderings
    orderings = [
        sorted(pool, reverse=True),
        sorted(pool),
        sorted(pool, key=lambda x: x if x > 128 else -x),  # large first, then small
        sorted(pool, key=lambda x: -x if x % 2 == 0 else x),  # even large first
    ]
    
    for ordering in orderings:
        if _remaining() < 3.0:
            break
        result = _greedy_construct(ordering, deadline_check=_remaining)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Random restarts
    if _remaining() > 10.0:
        for seed in range(50):
            if _remaining() < 3.0:
                break
            random.seed(seed)
            random_order = list(pool)
            random.shuffle(random_order)
            result = _greedy_construct(random_order, deadline_check=_remaining)
            if len(result) > best_len:
                best = result
                best_len = len(result)
    
    # Simple local search: try adding one element at a time
    if _remaining() > 5.0 and best:
        changed = True
        while changed and _remaining() > 3.0:
            changed = False
            current_sums = _subset_sums_from_list(best)
            for x in pool:
                if x in best:
                    continue
                new_sums = set(current_sums)
                new_sums.add(x)
                for s in current_sums:
                    new_sums.add(s + x)
                if _check_no_divisibility(new_sums):
                    best.append(x)
                    best_len += 1
                    current_sums = new_sums
                    changed = True
                    break
    
    # Final validation
    if best and not _validate_full(best):
        best = [128, 192, 224]
    
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
