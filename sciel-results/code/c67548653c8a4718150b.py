import random
import time

N = 256

def _compute_sums(selected):
    """Compute all unordered pair sums (including a+a) for a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            sums.add(a + selected[j])
    return sums

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered pair sums distinct)."""
    if len(selected) != len(set(selected)):
        return False
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is inclusion-maximal."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _get_unblocked(selected, sums):
    """Return list of missing elements that can be added (not blocked)."""
    sel_set = set(selected)
    unblocked = []
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            unblocked.append(x)
    return unblocked

def _try_add(selected, sums, x):
    """Try to add x to selected. Return (success, new_selected, new_sums)."""
    if 2 * x in sums:
        return False, selected, sums
    for a in selected:
        if x + a in sums:
            return False, selected, sums
    new_sums = set(sums)
    new_sums.add(2 * x)
    for a in selected:
        new_sums.add(x + a)
    new_selected = selected + [x]
    return True, new_selected, new_sums

def _blocking_count(selected, sums, x):
    """Count how many unblocked elements x would block if added."""
    sel_set = set(selected)
    count = 0
    for y in range(1, N + 1):
        if y in sel_set or y == x:
            continue
        # Check if y is currently unblocked
        if 2 * y in sums:
            continue
        y_blocked = False
        for a in selected:
            if y + a in sums:
                y_blocked = True
                break
        if y_blocked:
            continue
        # y is unblocked. Check if adding x would block y
        if 2 * y in sums or 2 * y == 2 * x:
            count += 1
            continue
        for a in selected:
            if y + a in sums:
                count += 1
                break
        else:
            if y + x in sums:
                count += 1
    return count

def _greedy_maximize(selected, sums, rng, deadline):
    """Greedily add elements until maximal, preferring those that block the most."""
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        unblocked = _get_unblocked(selected, sums)
        if not unblocked:
            break
        
        # Choose element to add: prefer one that blocks the most other unblocked elements
        best_x = None
        best_score = -1
        for x in unblocked:
            score = _blocking_count(selected, sums, x)
            if score > best_score:
                best_score = score
                best_x = x
        
        if best_x is None:
            best_x = rng.choice(unblocked)
        
        ok, selected, sums = _try_add(selected, sums, best_x)
        if not ok:
            break
    
    return selected, sums

def _swap_search(selected, sums, rng, deadline):
    """Try swapping elements to reduce size while maintaining validity and maximality."""
    selected = list(selected)
    sums = set(sums)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            for j in range(len(selected)):
                if i == j:
                    continue
                # Try removing selected[i] and adding selected[j] is not useful
                # Instead, try removing selected[i] and see if still maximal
                candidate = selected[:i] + selected[i+1:]
                if not candidate:
                    continue
                new_sums = _compute_sums(candidate)
                if not _is_valid_sidon(candidate):
                    continue
                if _is_maximal(candidate, new_sums):
                    selected = candidate
                    sums = new_sums
                    improved = True
                    break
            if improved:
                break
    
    return selected, sums

def _reduce_by_removal(selected, sums, rng, deadline):
    """Aggressively try to remove elements while maintaining maximality."""
    selected = list(selected)
    sums = set(sums)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            new_sums = _compute_sums(candidate)
            if not _is_valid_sidon(candidate):
                continue
            if _is_maximal(candidate, new_sums):
                selected = candidate
                sums = new_sums
                improved = True
                break
    
    return selected, sums

def _construct_greedy_set_cover(rng, deadline):
    """
    Construct a maximal Sidon set using a set-cover-like greedy approach.
    Each element x we select 'covers' (blocks) certain missing elements.
    We want to minimize the number of selected elements.
    """
    selected = []
    sums = set()
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    # Start with a small random seed
    for x in candidates:
        if len(selected) >= 12:
            break
        ok, selected, sums = _try_add(selected, sums, x)
    
    # Now greedily maximize
    selected, sums = _greedy_maximize(selected, sums, rng, deadline)
    
    return selected, sums

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Multiple strategies with different random seeds
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Strategy: Random initial Sidon set, then greedy maximize, then reduce
        selected, sums = _construct_greedy_set_cover(rng, deadline)
        
        if not selected:
            continue
        
        if not _is_valid_sidon(selected):
            continue
        if not _is_maximal(selected, sums):
            continue
        
        if len(selected) < best_size:
            best_size = len(selected)
            best = list(selected)
        
        # Try to reduce further
        if time.time() < deadline:
            reduced, reduced_sums = _reduce_by_removal(selected, sums, rng, deadline)
            if _is_valid_sidon(reduced) and _is_maximal(reduced, reduced_sums):
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best = list(reduced)
        
        if time.time() < deadline:
            swapped, swapped_sums = _swap_search(reduced, reduced_sums, rng, deadline)
            if _is_valid_sidon(swapped) and _is_maximal(swapped, swapped_sums):
                if len(swapped) < best_size:
                    best_size = len(swapped)
                    best = list(swapped)
        
        # Early termination if we find a very small set
        if best_size <= 10:
            break
    
    # Fallback: deterministic greedy
    if best is None:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok, selected, sums = _try_add(selected, sums, x)
        return selected
    
    return best
