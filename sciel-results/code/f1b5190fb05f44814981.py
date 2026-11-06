"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _compute_sums_set(selected):
    """Compute all pairwise sums a+b for a<=b in selected as a set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_sidon_fast(selected):
    """Fast Sidon check using numpy."""
    arr = np.array(selected, dtype=np.int32)
    n = len(arr)
    if n == 0:
        return True
    total = n * (n + 1) // 2
    sums = np.zeros(total, dtype=np.int32)
    idx = 0
    for i in range(n):
        sums[idx] = 2 * arr[i]
        idx += 1
        for j in range(i + 1, n):
            sums[idx] = arr[i] + arr[j]
            idx += 1
    return len(np.unique(sums)) == total

def _can_add(x, selected, sums):
    """Check if x can be added to selected while maintaining Sidon property."""
    if 2 * x in sums:
        return False
    for s in selected:
        if x + s in sums:
            return False
    # Also check internal consistency: x+x vs x+s (already covered by 2x check)
    # and x+s1 vs x+s2 (s1!=s2 so x+s1!=x+s2)
    return True

def _is_maximal_fast(selected):
    """Check if selected is inclusion-maximal Sidon in [1,N]."""
    selected_set = set(selected)
    sums = _compute_sums_set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if not _can_add(x, selected, sums):
            continue
        return False
    return True

def _get_addable(selected, sums):
    """Return set of elements that can be added to selected."""
    selected_set = set(selected)
    addable = set()
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if _can_add(x, selected, sums):
            addable.add(x)
    return addable

def _construct_maximal_greedy(order, initial=None):
    """Greedy Sidon set construction following the given order."""
    if initial is not None:
        selected = list(initial)
        sums = _compute_sums_set(selected)
    else:
        selected = []
        sums = set()
    selected_set = set(selected)
    for x in order:
        if x in selected_set:
            continue
        if _can_add(x, selected, sums):
            selected.append(x)
            sums.add(2 * x)
            for s in selected[:-1]:
                sums.add(x + s)
            selected_set.add(x)
    return selected

def _lazy_deletion_reduce(selected, deadline, rng=None):
    """
    Structural change: lazy deletion / minimum vertex cover on dependency graph.
    
    Instead of just checking if removal maintains maximality, we model the
    dependency structure: for each element x in the set, determine which
    addable elements x 'blocks'. Then we try to remove elements that block
    the fewest addable elements, and check if the remaining set is still
    maximal (i.e., the removed element's blocked elements are all blocked
    by other remaining elements).
    """
    if rng is None:
        rng = random.Random()
    
    current = list(selected)
    improved = True
    pass_count = 0
    max_passes = 100
    
    while improved and time.time() < deadline and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        sums = _compute_sums_set(current)
        current_set = set(current)
        addable = _get_addable(current, sums)
        
        if not addable:
            break
        
        # For each element x in current, compute which addable elements it blocks
        # An addable element y is 'blocked by' x if removing x would make y addable
        # More precisely: y is addable w.r.t. current\{x} iff it's not addable w.r.t. current
        # But we need: after removing x, is y still not addable?
        # y is not addable to current means there exists some conflict.
        # After removing x, y might become addable if the only conflict was with x.
        
        # Build dependency: for each x in current, which addable y would become
        # addable if x were removed?
        removal_impact = {}
        for x in current:
            without_x = [y for y in current if y != x]
            if not without_x:
                removal_impact[x] = set()
                continue
            sums_x = _compute_sums_set(without_x)
            # Which previously-addable elements become addable after removing x?
            # Actually, we want: which elements that are NOT addable to current
            # WOULD become addable to current\{x}?
            # That tells us what x is "protecting"
            newly_addable = set()
            for y in range(1, N + 1):
                if y in current_set:
                    continue
                # y is not addable to current (by definition of addable set)
                # Check if y is addable to without_x
                if _can_add(y, without_x, sums_x):
                    newly_addable.add(y)
            removal_impact[x] = newly_addable
        
        # Try removing the element with smallest impact
        best_x = None
        best_impact_size = float('inf')
        candidates = []
        for x in current:
            impact_size = len(removal_impact[x])
            if impact_size < best_impact_size:
                best_impact_size = impact_size
                best_x = x
                candidates = [x]
            elif impact_size == best_impact_size:
                candidates.append(x)
        
        if not candidates:
            break
        
        # Try removing a candidate (randomly chosen from best)
        x_to_remove = rng.choice(candidates)
        new_current = [y for y in current if y != x_to_remove]
        if not new_current:
            break
        
        # Verify the new set is still maximal
        if _is_maximal_fast(new_current):
            current = new_current
            improved = True
        else:
            # Try other candidates
            found = False
            for alt in candidates:
                if alt == x_to_remove:
                    continue
                alt_current = [y for y in current if y != alt]
                if not alt_current:
                    continue
                if _is_maximal_fast(alt_current):
                    current = alt_current
                    improved = True
                    found = True
                    break
            if not found:
                # Try swaps: remove x, add some y to maintain maximality
                for x in current:
                    without_x = [y for y in current if y != x]
                    if not without_x:
                        continue
                    sums_x = _compute_sums_set(without_x)
                    for y in range(1, N + 1):
                        if y in set(without_x):
                            continue
                        if not _can_add(y, without_x, sums_x):
                            continue
                        candidate = without_x + [y]
                        if _is_maximal_fast(candidate):
                            current = candidate
                            improved = True
                            found = True
                            break
                    if found:
                        break
            if not found:
                break
    
    return current

def _construct_with_minimal_core(seed_val, deadline):
    """
    Construct a maximal Sidon set by:
    1. Building a very small core (size 3-8)
    2. Using set-cover style forced additions to make it maximal
    3. Applying lazy deletion to minimize size
    """
    rng = random.Random(seed_val)
    
    # Try to build a small core
    core_size = rng.randint(3, 8)
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    core = []
    core_sums = set()
    for x in candidates:
        if len(core) >= core_size:
            break
        if _can_add(x, core, core_sums):
            core.append(x)
            core_sums.add(2 * x)
            for s in core[:-1]:
                core_sums.add(x + s)
    
    if len(core) < 3:
        return None
    
    # Now force maximality: repeatedly add the element that blocks the most addable elements
    selected = list(core)
    sums = _compute_sums_set(selected)
    selected_set = set(selected)
    
    max_iter = 500
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        addable = _get_addable(selected, sums)
        if not addable:
            break
        
        # For each candidate that can be added, compute how many addable elements it blocks
        best_candidate = None
        best_blocked = -1
        best_candidates = []
        
        for cand in range(1, N + 1):
            if cand in selected_set:
                continue
            if not _can_add(cand, selected, sums):
                continue
            
            # Simulate adding cand
            new_selected = selected + [cand]
            new_sums = sums | {cand + a for a in selected} | {2 * cand}
            
            # Count how many currently-addable elements are blocked
            blocked = 0
            for y in addable:
                if y == cand:
                    continue
                if not _can_add(y, new_selected, new_sums):
                    blocked += 1
            
            if blocked > best_blocked:
                best_blocked = blocked
                best_candidates = [cand]
            elif blocked == best_blocked:
                best_candidates.append(cand)
        
        if not best_candidates:
            break
        
        best_candidate = rng.choice(best_candidates)
        selected.append(best_candidate)
        selected_set.add(best_candidate)
        sums = _compute_sums_set(selected)
    
    # Verify
    if not _is_sidon_fast(selected):
        return None
    if not _is_maximal_fast(selected):
        return None
    
    # Apply lazy deletion
    reduced = _lazy_deletion_reduce(selected, deadline, rng=random.Random(seed_val * 99991))
    
    if _is_sidon_fast(reduced) and _is_maximal_fast(reduced):
        return reduced
    return selected

def _construct_random_permutation(seed_val, deadline):
    """
    Construct maximal Sidon set by simulating greedy addition of a random permutation,
    then applying lazy deletion.
    """
    rng = random.Random(seed_val)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    
    selected = _construct_maximal_greedy(order)
    
    if not selected:
        return None
    
    if not _is_sidon_fast(selected):
        return None
    
    # The greedy construction gives a maximal set (by construction, we add all compatible elements)
    # But it might not be inclusion-maximal in the strong sense - let's verify
    if not _is_maximal_fast(selected):
        # Force maximality
        sums = _compute_sums_set(selected)
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if _can_add(x, selected, sums):
                selected.append(x)
                sums.add(2 * x)
                for s in selected[:-1]:
                    sums.add(x + s)
                selected_set.add(x)
    
    # Apply lazy deletion
    reduced = _lazy_deletion_reduce(selected, deadline, rng=rng)
    
    if _is_sidon_fast(reduced) and _is_maximal_fast(reduced):
        return reduced
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    best = None
    best_size = N + 1
    
    # Strategy 1: Baseline ascending greedy + lazy deletion
    order = list(range(1, N + 1))
    selected = _construct_maximal_greedy(order)
    if selected and _is_sidon_fast(selected):
        if not _is_maximal_fast(selected):
            sums = _compute_sums_set(selected)
            selected_set = set(selected)
            for x in range(1, N + 1):
                if x not in selected_set and _can_add(x, selected, sums):
                    selected.append(x)
                    sums.add(2 * x)
                    for s in selected[:-1]:
                        sums.add(x + s)
                    selected_set.add(x)
        reduced = _lazy_deletion_reduce(selected, deadline, rng=random.Random(1))
        if _is_sidon_fast(reduced) and _is_maximal_fast(reduced):
            best = reduced
            best_size = len(reduced)
    
    # Strategy 2: Minimal core + forced additions + lazy deletion (many trials)
    if time.time() < deadline - 5.0:
        for trial in range(400):
            if time.time() > deadline:
                break
            result = _construct_with_minimal_core(trial, deadline)
            if result is None:
                continue
            if len(result) < best_size:
                best = result
                best_size = len(result)
    
    # Strategy 3: Random permutations + lazy deletion
    if time.time() < deadline - 3.0:
        for trial in range(200):
            if time.time() > deadline:
                break
            result = _construct_random_permutation(trial * 104729 + 7, deadline)
            if result is None:
                continue
            if len(result) < best_size:
                best = result
                best_size = len(result)
    
    # Strategy 4: More aggressive lazy deletion on best result
    if best is not None and time.time() < deadline - 2.0:
        for trial in range(50):
            if time.time() > deadline:
                break
            reduced = _lazy_deletion_reduce(best, deadline, rng=random.Random(trial * 31337 + 5))
            if _is_sidon_fast(reduced) and _is_maximal_fast(reduced):
                if len(reduced) < best_size:
                    best = reduced
                    best_size = len(reduced)
    
    # Final safety check
    if best is None:
        best = _construct_maximal_greedy(list(range(1, N + 1)))
    
    return best
# EVOLVE-BLOCK-END
