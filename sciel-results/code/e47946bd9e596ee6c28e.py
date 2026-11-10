"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 256

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check if selected is a valid Sidon set."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    for a in selected:
        for b in selected:
            if a <= b:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
    return True

def _is_maximal(selected, N):
    """Check if selected is inclusion-maximal in [1,N]."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Try adding x
        new_selected = selected + [x]
        if _is_sidon(new_selected):
            return False
    return True

def _validate_and_get_size(selected):
    """Return (is_valid_maximal, size) for a candidate set."""
    if not selected:
        return False, 0
    if not _is_sidon(selected):
        return False, len(selected)
    if not _is_maximal(selected, N):
        return False, len(selected)
    return True, len(selected)

def _greedy_construct(start_order=None, deadline=None):
    """
    Greedy construction: iteratively pick the element that blocks the most
    other elements from being added, while maintaining Sidon property.
    """
    if deadline is None:
        deadline = time.time() + 55  # safety margin
    
    if start_order is None:
        start_order = list(range(1, N + 1))
    
    selected = []
    selected_set = set()
    sums = set()
    
    for x in start_order:
        if time.time() > deadline:
            break
        
        # Check if adding x maintains Sidon
        added_sums = set()
        valid = True
        for a in selected:
            s = x + a
            if s in sums or s in added_sums:
                valid = False
                break
            added_sums.add(s)
        if not valid:
            continue
        s = 2 * x
        if s in sums or s in added_sums:
            continue
        added_sums.add(s)
        
        selected.append(x)
        selected_set.add(x)
        sums.update(added_sums)
    
    return selected

def _try_remove_to_improve(selected, deadline=None):
    """
    Try to reduce the size of a valid maximal Sidon set by removing elements
    and checking if the result is still maximal. If not maximal, we can't remove.
    So this function tries to find a smaller maximal Sidon set by local search.
    """
    if deadline is None:
        deadline = time.time() + 55
    
    current = list(selected)
    current_set = set(current)
    
    # Try removing each element one by one; if the set is no longer maximal,
    # we must keep it. But maybe we can remove one and add another to get smaller?
    # Actually, for inclusion-maximal, we can't just remove. We need a different approach.
    
    # Local search: try swapping pairs to see if we can get a smaller maximal set.
    # This is complex. Instead, let's try a different construction strategy.
    
    return current

def _randomized_greedy(seed_val, deadline=None):
    """
    Randomized greedy: shuffle the order, then greedily construct.
    """
    if deadline is None:
        deadline = time.time() + 55
    
    rng = random.Random(seed_val)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    
    selected = []
    sums = set()
    
    for x in order:
        if time.time() > deadline:
            break
        
        added_sums = set()
        valid = True
        for a in selected:
            s = x + a
            if s in sums or s in added_sums:
                valid = False
                break
            added_sums.add(s)
        if not valid:
            continue
        s = 2 * x
        if s in sums or s in added_sums:
            continue
        added_sums.add(s)
        
        selected.append(x)
        sums.update(added_sums)
    
    return selected

def _maximality_check_and_fix(selected, deadline=None):
    """
    Verify maximality. If not maximal, this shouldn't happen from greedy,
    but let's verify.
    """
    return _is_maximal(selected, N)

def construct():
    """
    Find a small inclusion-maximal Sidon set in [1, 256].
    """
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    best_set = None
    best_size = N + 1
    
    # First, try the standard ascending greedy to get a baseline
    baseline = _greedy_construct(list(range(1, N + 1)), deadline)
    is_valid, size = _validate_and_get_size(baseline)
    if is_valid:
        best_set = baseline
        best_size = size
    
    # Now try randomized greedy with multiple seeds
    # The key insight: different orderings may lead to smaller maximal sets
    num_trials = 200
    for i in range(num_trials):
        if time.time() > deadline:
            break
        
        trial_set = _randomized_greedy(i, deadline)
        if not trial_set:
            continue
        
        is_valid, size = _validate_and_get_size(trial_set)
        if is_valid and size < best_size:
            best_set = trial_set
            best_size = size
    
    # Also try reverse order
    if time.time() < deadline:
        rev_set = _greedy_construct(list(range(N, 0, -1)), deadline)
        if rev_set:
            is_valid, size = _validate_and_get_size(rev_set)
            if is_valid and size < best_size:
                best_set = rev_set
                best_size = size
    
    # Try to improve by local search: try removing elements and see if we
    # can find a smaller maximal set. For a maximal Sidon set, we can't remove
    # any element without breaking maximality (by definition). But we can try
    # different constructions.
    
    if best_set is None:
        # Fallback: return the baseline
        best_set = baseline
    
    return best_set
# EVOLVE-BLOCK-END
