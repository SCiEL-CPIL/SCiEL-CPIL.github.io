"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _build_sum_mask(selected):
    """Build an integer bitmask of all sums a+b for a<=b in selected."""
    mask = 0
    n = len(selected)
    for i in range(n):
        a = selected[i]
        mask |= (1 << (2 * a))
        for j in range(i + 1, n):
            s = a + selected[j]
            mask |= (1 << s)
    return mask

def _is_valid_sidon(selected):
    """Check if selected is a Sidon set."""
    mask = 0
    n = len(selected)
    for i in range(n):
        a = selected[i]
        bit = 1 << (2 * a)
        if mask & bit:
            return False
        mask |= bit
        for j in range(i + 1, n):
            s = a + selected[j]
            bit = 1 << s
            if mask & bit:
                return False
            mask |= bit
    return True

def _is_maximal(selected, N, sum_mask=None):
    """Check if selected is inclusion-maximal."""
    if sum_mask is None:
        sum_mask = _build_sum_mask(selected)
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x is addable
        if (sum_mask >> (2 * x)) & 1:
            continue
        ok = True
        for a in selected:
            if (sum_mask >> (x + a)) & 1:
                ok = False
                break
        if ok:
            return False
    return True

def _addable(x, selected, sum_mask):
    """Can x be added to selected (with current sum mask) without conflict?"""
    if (sum_mask >> (2 * x)) & 1:
        return False
    for a in selected:
        if (sum_mask >> (x + a)) & 1:
            return False
    return True

def _max_blocking_greedy(N, deadline, seed=None):
    rng = random.Random(seed) if seed is not None else None
    remaining = list(range(1, N + 1))
    selected = []
    sum_mask = 0
    while remaining:
        if time.time() > deadline:
            break
        valid = [x for x in remaining if _addable(x, selected, sum_mask)]
        if not valid:
            break
        best_x = None
        best_block = -1
        for x in valid:
            # Calculate new mask if x added
            new_mask = sum_mask | (1 << (2 * x))
            for a in selected:
                new_mask |= (1 << (x + a))
            
            block = 0
            for y in valid:
                if y == x:
                    continue
                # Check if y is blocked by new_mask
                if (new_mask >> (2 * y)) & 1:
                    block += 1
                    continue
                blocked = False
                for a in selected:
                    if (new_mask >> (y + a)) & 1:
                        blocked = True
                        break
                if not blocked and (new_mask >> (x + y)) & 1:
                    blocked = True
                if blocked:
                    block += 1
            if block > best_block:
                best_block = block
                best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        remaining.remove(best_x)
        sum_mask |= (1 << (2 * best_x))
        for a in selected[:-1]:
            sum_mask |= (1 << (best_x + a))
    return selected

def _coverage_greedy_fill(selected, N, deadline, seed=None):
    rng = random.Random(seed) if seed is not None else None
    selected = list(selected)
    sum_mask = _build_sum_mask(selected)
    sel_set = set(selected)
    remaining = [x for x in range(1, N + 1) if x not in sel_set]
    while remaining:
        if time.time() > deadline:
            break
        valid = [x for x in remaining if _addable(x, selected, sum_mask)]
        if not valid:
            break
        best_x = None
        best_cov = -1
        for x in valid:
            new_mask = sum_mask | (1 << (2 * x))
            for a in selected:
                new_mask |= (1 << (x + a))
            
            cov = 0
            for y in remaining:
                if y == x:
                    continue
                if (new_mask >> (2 * y)) & 1:
                    cov += 1
                    continue
                blocked = False
                for a in selected:
                    if (new_mask >> (y + a)) & 1:
                        blocked = True
                        break
                if not blocked and (new_mask >> (x + y)) & 1:
                    blocked = True
                if blocked:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        sel_set.add(best_x)
        remaining.remove(best_x)
        sum_mask |= (1 << (2 * best_x))
        for a in selected[:-1]:
            sum_mask |= (1 << (best_x + a))
    return selected

def _local_search_remove(selected, N, deadline):
    best = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            if time.time() > deadline:
                break
            cand = [v for v in best if v != x]
            if not _is_valid_sidon(cand):
                continue
            # Rebuild mask for cand to check maximality efficiently
            cand_mask = _build_sum_mask(cand)
            if _is_maximal(cand, N, cand_mask):
                best = cand
                improved = True
                break
    return best

def _random_greedy_seed(N, deadline, seed):
    rng = random.Random(seed)
    remaining = list(range(1, N + 1))
    selected = []
    sum_mask = 0
    while remaining:
        if time.time() > deadline:
            break
        valid = [x for x in remaining if _addable(x, selected, sum_mask)]
        if not valid:
            break
        best_x = None
        best_cov = -1
        top = []
        for x in valid:
            new_mask = sum_mask | (1 << (2 * x))
            for a in selected:
                new_mask |= (1 << (x + a))
            
            cov = 0
            for y in valid:
                if y == x:
                    continue
                if (new_mask >> (2 * y)) & 1:
                    cov += 1
                    continue
                blocked = False
                for a in selected:
                    if (new_mask >> (y + a)) & 1:
                        blocked = True
                        break
                if not blocked and (new_mask >> (x + y)) & 1:
                    blocked = True
                if blocked:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                top = [x]
            elif cov == best_cov:
                top.append(x)
        x = rng.choice(top) if top else rng.choice(valid)
        selected.append(x)
        remaining.remove(x)
        sum_mask |= (1 << (2 * x))
        for a in selected[:-1]:
            sum_mask |= (1 << (x + a))
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    best_result = None
    best_size = float('inf')

    def consider(selected):
        nonlocal best_result, best_size
        if not selected:
            return
        if not _is_valid_sidon(selected):
            return
        # Use bitset for maximality check
        mask = _build_sum_mask(selected)
        if not _is_maximal(selected, N, mask):
            return
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = list(selected)

    # Strategy A: max-blocking greedy -> complete to maximal -> local search shrink
    for seed in range(12):
        if time.time() > deadline:
            break
        partial = _max_blocking_greedy(N, deadline, seed if seed else None)
        if not partial:
            continue
        maximal = _coverage_greedy_fill(partial, N, deadline, seed)
        consider(maximal)
        shrunken = _local_search_remove(maximal, N, deadline)
        consider(shrunken)

    # Strategy B: randomized seeds -> complete -> shrink
    for seed in range(24):
        if time.time() > deadline:
            break
        partial = _random_greedy_seed(N, deadline, 1000 + seed)
        if not partial:
            continue
        maximal = _coverage_greedy_fill(partial, N, deadline, seed)
        consider(maximal)
        shrunken = _local_search_remove(maximal, N, deadline)
        consider(shrunken)

    # Fallback: ascending greedy, then make maximal
    if best_result is None:
        selected = []
        sum_mask = 0
        for x in range(1, N + 1):
            if _addable(x, selected, sum_mask):
                selected.append(x)
                sum_mask |= (1 << (2 * x))
                for a in selected[:-1]:
                    sum_mask |= (1 << (x + a))
        maximal = _coverage_greedy_fill(selected, N, deadline, 0)
        consider(maximal)
        if best_result is None:
            return maximal

    return best_result

# EVOLVE-BLOCK-END
