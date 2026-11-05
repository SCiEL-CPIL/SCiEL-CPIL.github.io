"""Fast integer-based meet-in-the-middle for reciprocal subset-sum checks.
Uses LCM-scaled integer arithmetic to avoid slow Fraction operations.
Combined with systematic search by cardinality for maximum set size."""
import time
import sys
import math
import random

# --- Precomputation with integer arithmetic ---
_LCM_24 = 5354228880  # lcm(1,2,...,24)

# For each element a (1..24), 1/a scaled by LCM_24:
_REC = [0] + [_LCM_24 // a for a in range(1, 25)]  # index 1..24

# Left half: elements 1..12, Right half: elements 13..24
_LEFT = list(range(1, 13))   # elements 1..12
_RIGHT = list(range(13, 25))  # elements 13..24

def _build_int_subset_sums(arr):
    """Build dict: integer_sum -> list of masks (bitmask into arr) for nonempty subsets.
    Also includes empty subset: sum=0 -> [0]."""
    n = len(arr)
    sums = {0: [0]}  # include empty
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask >> i & 1:
                s += _REC[arr[i]]
        if s not in sums:
            sums[s] = []
        sums[s].append(mask)
    return sums

_t0 = time.time()
_sums_left = _build_int_subset_sums(_LEFT)
_sums_right = _build_int_subset_sums(_RIGHT)

# Build violation database: for each target a (1..24), store all (lmask, rmask) pairs
# where lmask is a mask into _LEFT (12 bits) and rmask is a mask into _RIGHT (12 bits)
# such that sum_left(lmask) + sum_right(rmask) == _REC[a], and not both zero.
_VIOLATION_DB = {}  # a (1..24) -> list of (lmask, rmask)

for a in range(1, 25):
    target = _REC[a]
    pairs = []
    for sL, l_masks in _sums_left.items():
        needR = target - sL
        if needR < 0:
            continue
        r_masks = _sums_right.get(needR, [])
        for lm in l_masks:
            for rm in r_masks:
                if lm == 0 and rm == 0:
                    continue
                pairs.append((lm, rm))
    _VIOLATION_DB[a] = pairs

def _check_safe(a_idx, S_mask):
    """Check if element a_idx (0-based, element = a_idx+1) in set S_mask is safe.
    S_mask: 24-bit int where bit i means element (i+1) is in S.
    Returns True if no nonempty subset of S \\ {a_idx} sums to 1/(a_idx+1).
    """
    val = a_idx + 1
    avail_mask = S_mask & ~(1 << a_idx)
    
    if avail_mask == 0:
        return True
    
    left_avail = avail_mask & 0xFFF       # bits 0..11 (elements 1..12)
    right_avail = (avail_mask >> 12) & 0xFFF  # bits 12..23 mapped to 0..11 (elements 13..24)
    
    pairs = _VIOLATION_DB.get(val, [])
    if not pairs:
        return True
    
    for lm, rm in pairs:
        if (lm & left_avail) == lm and (rm & right_avail) == rm:
            return False
    return True

def _check_all_safe(S_mask):
    """Check all elements in S_mask are safe."""
    for i in range(24):
        if S_mask >> i & 1:
            if not _check_safe(i, S_mask):
                return False
    return True

def _remove_violations(S_mask):
    """Remove elements from S_mask until no violations remain."""
    while S_mask:
        if _check_all_safe(S_mask):
            break
        violators = []
        for i in range(24):
            if S_mask >> i & 1:
                if not _check_safe(i, S_mask):
                    violators.append(i)
        if not violators:
            break
        to_remove = min(violators)
        S_mask &= ~(1 << to_remove)
    return S_mask

def _greedy_build(order, deadline):
    """Greedy build in given order."""
    S = 0
    for idx in order:
        if time.time() > deadline:
            break
        trial = S | (1 << idx)
        if _check_all_safe(trial):
            S = trial
    return S

def _local_search_add(S, deadline):
    """Try to add more elements to S."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(24):
            if time.time() > deadline:
                break
            if S >> idx & 1:
                continue
            trial = S | (1 << idx)
            if _check_all_safe(trial):
                S = trial
                improved = True
    return S

def _swap_search(S, deadline):
    """Try swaps to enable further additions."""
    best = S
    best_size = S.bit_count()
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for out_idx in range(24):
            if time.time() > deadline:
                break
            if not (S >> out_idx & 1):
                continue
            for in_idx in range(24):
                if time.time() > deadline:
                    break
                if S >> in_idx & 1:
                    continue
                trial = S & ~(1 << out_idx) | (1 << in_idx)
                if _check_all_safe(trial):
                    S2 = trial
                    S2 = _local_search_add(S2, deadline)
                    if S2.bit_count() > best_size:
                        best = S2
                        best_size = S2.bit_count()
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return best

def construct():
    """Return a nonempty subset of [1,24] maximizing cardinality such that no reciprocal
    of a selected element equals the sum of reciprocals of a subset of the other selected elements."""
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill
    
    best_set = 0
    best_size = 0
    
    # Phase 1: Multiple greedy orderings
    orderings = [
        list(range(24)),
        list(range(23, -1, -1)),
        [i for i in range(24) if i % 2 == 0] + [i for i in range(24) if i % 2 == 1],
        [i for i in range(24) if i % 2 == 1] + [i for i in range(24) if i % 2 == 0],
        [i for i in range(24) if i+1 in (2,3,5,7,11,13,17,19,23)] + 
        [i for i in range(24) if i+1 not in (2,3,5,7,11,13,17,19,23)],
        # Try starting with large elements (small reciprocals, less likely to be targets)
        [i for i in range(23, -1, -1) if i+1 in (2,3,5,7,11,13,17,19,23)] + 
        [i for i in range(23, -1, -1) if i+1 not in (2,3,5,7,11,13,17,19,23)],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_build(order, deadline)
        S = _local_search_add(S, deadline)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_set = S
    
    # Phase 2: Randomized restarts
    rng = random.Random(42)
    for attempt in range(500):
        if time.time() > deadline:
            break
        # Random subset
        S = 0
        for idx in range(24):
            if rng.random() < 0.5:
                S |= (1 << idx)
        S = _remove_violations(S)
        S = _local_search_add(S, deadline)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_set = S
    
    # Phase 3: Swap-based improvement
    if time.time() < deadline:
        S = _swap_search(best_set, deadline)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_set = S
    
    # Phase 4: Try to find larger sets via targeted search
    # For each possible size from best_size+1 up, try to find a valid set
    if time.time() < deadline:
        for target_size in range(best_size + 1, 25):
            if time.time() > deadline:
                break
            # Try random combinations of target_size
            found = False
            for attempt in range(200):
                if time.time() > deadline:
                    break
                # Random subset of target_size
                indices = rng.sample(range(24), target_size)
                S = 0
                for idx in indices:
                    S |= (1 << idx)
                S = _remove_violations(S)
                S = _local_search_add(S, deadline)
                if S.bit_count() >= target_size:
                    if S.bit_count() > best_size:
                        best_size = S.bit_count()
                        best_set = S
                        found = True
                        break
            if found:
                break
    
    # Convert best_set to list
    if best_set:
        result = [i + 1 for i in range(24) if best_set >> i & 1]
    else:
        result = [1]
    
    if not result:
        result = [1]
    
    return result

# Test
if __name__ == "__main__":
    t0 = time.time()
    r = construct()
    t1 = time.time()
    print(f"Result: {r}, size={len(r)}, time={t1-t0:.2f}s")
