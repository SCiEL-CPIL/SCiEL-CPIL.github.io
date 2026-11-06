"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

TARGET = 200
MAXV = 100

def _feasible_check(selected):
    """Return True if no subset of selected sums to TARGET (each element at most once)."""
    reachable = {0}
    for x in selected:
        # Only sums up to TARGET matter
        new = {s + x for s in reachable if s + x <= TARGET}
        if TARGET in new:
            return False
        reachable |= new
    return True

def _greedy_descending():
    """Greedy: add largest elements that don't create a subset sum of 200."""
    selected = []
    reachable = {0}
    for x in range(MAXV, 0, -1):
        new = {s + x for s in reachable if s + x <= TARGET}
        if TARGET not in new:
            selected.append(x)
            reachable |= new
    return selected

def _greedy_ascending():
    """Greedy: add smallest elements that don't create a subset sum of 200."""
    selected = []
    reachable = {0}
    for x in range(1, MAXV + 1):
        new = {s + x for s in reachable if s + x <= TARGET}
        if TARGET not in new:
            selected.append(x)
            reachable |= new
    return selected

def _local_search_improve(selected, deadline):
    """
    Try to improve cardinality via local search:
    - Remove elements to allow adding others
    - Try all single removals, then try adding all remaining elements
    """
    selected = list(selected)
    best = list(selected)
    best_size = len(selected)
    
    removed = set(selected)
    remaining = [x for x in range(1, MAXV + 1) if x not in removed]
    
    # Try each single removal
    for i in range(len(selected)):
        if time.time() > deadline:
            break
        trial = list(selected[:i] + selected[i+1:])
        # Try to add back all remaining elements greedily
        reachable = {0}
        for x in sorted(trial):
            new = {s + x for s in reachable if s + x <= TARGET}
            reachable |= new
        # Now try to add each remaining element
        new_size = len(trial)
        for x in remaining:
            if x in trial:
                continue
            new = {s + x for s in reachable if s + x <= TARGET}
            if TARGET not in new:
                trial.append(x)
                reachable |= new
                new_size += 1
        if new_size > best_size:
            best = trial
            best_size = new_size
            removed = set(best)
            remaining = [x for x in range(1, MAXV + 1) if x not in removed]
    
    return best

def _bitmask_subset_sum_check(selected):
    """
    Use a different approach: represent reachable sums as a bitset.
    Bit i is 1 if sum i is reachable.
    """
    # bits: bit 0 = sum 0, bit k = sum k
    bits = 1  # only sum 0 reachable
    for x in selected:
        # shift left by x, mask to TARGET bits
        shifted = (bits << x) & ((1 << (TARGET + 1)) - 1)
        if shifted & (1 << TARGET):
            return False
        bits |= shifted
    return True

def _construct_with_bitmask():
    """Greedy descending using bitmask subset sum check."""
    selected = []
    bits = 1
    mask = (1 << (TARGET + 1)) - 1
    for x in range(MAXV, 0, -1):
        shifted = (bits << x) & mask
        if not (shifted & (1 << TARGET)):
            selected.append(x)
            bits |= shifted
    return selected

def construct():
    start = time.time()
    deadline = start + 80.0  # 80s safety margin (limit is 90s)
    
    # Generate multiple candidate solutions
    candidates = []
    
    # 1. Greedy descending (original)
    candidates.append(_greedy_descending())
    
    # 2. Greedy ascending
    candidates.append(_greedy_ascending())
    
    # 3. Greedy descending with bitmask (should be same as 1 but verify)
    candidates.append(_construct_with_bitmask())
    
    # Pick the best by cardinality
    best = max(candidates, key=len)
    
    # Verify feasibility
    if not _feasible_check(best):
        # Fallback to simplest valid
        best = _greedy_descending()
    
    # Try local search improvement
    if time.time() < deadline - 5.0:
        improved = _local_search_improve(best, deadline - 2.0)
        if len(improved) > len(best):
            if _feasible_check(improved):
                best = improved
    
    # Final verification
    if not _feasible_check(best):
        # Fallback: just return greedy descending (known valid)
        best = _greedy_descending()
    
    return best
# EVOLVE-BLOCK-END
