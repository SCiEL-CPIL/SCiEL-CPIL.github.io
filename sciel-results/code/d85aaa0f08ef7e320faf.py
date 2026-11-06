import time
import random

def _is_valid_mean_free(selected, candidate):
    """Check if adding candidate to selected keeps the set mean-free.
    
    For each element a in selected ∪ {candidate}, check that no subset of
    at least 2 OTHER elements has sum = len(subset) * a.
    
    Uses bitset DP: bitsets[j] is a Python int where bit s is set if sum s
    is achievable with exactly j elements from the 'others' list.
    """
    full = selected + [candidate]
    n = len(full)
    
    for a in full:
        others = [x for x in full if x != a]
        m = len(others)
        
        if m < 2:
            continue
        
        # Maximum target we need to check: k*a for k in [2, m], a <= 512
        target_max = m * 512
        mask = (1 << (target_max + 1)) - 1
        
        bitsets = [0] * (m + 1)
        bitsets[0] = 1  # sum 0 with 0 elements
        
        count = 0
        for e in others:
            for j in range(count + 1, 0, -1):
                bitsets[j] |= (bitsets[j-1] << e) & mask
            count += 1
        
        for k in range(2, m + 1):
            target = k * a
            if target <= target_max and (bitsets[k] >> target) & 1:
                return False
    
    return True

def _modular_pre_filter(selected, candidate):
    """Quick necessary-condition check for k=2 violations involving candidate."""
    sel_set = set(selected)
    
    # For a = candidate: no pair in selected sums to 2*candidate
    for s in selected:
        complement = 2 * candidate - s
        if complement in sel_set and complement != s:
            return False
    
    # For a in selected: no pair (candidate, s) sums to 2*a
    for a in selected:
        complement = 2 * a - candidate
        if complement in sel_set and complement != a and complement != candidate:
            return False
    
    return True

def _is_valid_mean_free_fast(selected, candidate):
    """Faster check using modular pre-filter first, then full bitset DP."""
    if not _modular_pre_filter(selected, candidate):
        return False
    return _is_valid_mean_free(selected, candidate)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_result = []
    best_len = 0
    
    # Multiple greedy passes with different orderings
    orderings = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
    ]
    
    random.seed(42)
    for _ in range(20):
        orderings.append(random.sample(range(1, 513), 512))
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        selected = []
        for x in ordering:
            if time.time() > deadline:
                break
            if _is_valid_mean_free_fast(selected, x):
                selected.append(x)
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_result = selected[:]
        
        if best_len >= 48:
            break
    
    return best_result
