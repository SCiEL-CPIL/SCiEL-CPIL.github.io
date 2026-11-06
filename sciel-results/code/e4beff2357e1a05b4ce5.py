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

def _greedy_maximize(selected, sums, rng, deadline):
    """Greedily add elements until maximal, trying to minimize additions."""
    selected = list(selected)
    sums = set(sums)
    
    while time.time() < deadline:
        unblocked = _get_unblocked(selected, sums)
        if not unblocked:
            break
        
        # If unblocked list is large, sample to speed up scoring
        if len(unblocked) > 40:
            candidates = rng.sample(unblocked, 40)
        else:
            candidates = list(unblocked)
            rng.shuffle(candidates)
            
        best_x = None
        max_blocks = -1
        ties = []
        
        for x in candidates:
            new_sums = set(sums)
            valid = True
            if 2 * x in new_sums:
                valid = False
            else:
                new_sums.add(2 * x)
                for a in selected:
                    s = x + a
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
            if not valid:
                continue
            
            score = 0
            for y in unblocked:
                if y == x:
                    continue
                if 2 * y in new_sums:
                    score += 1
                    continue
                y_blocked = False
                for a in selected:
                    if y + a in new_sums:
                        y_blocked = True
                        break
                if not y_blocked:
                    if y + x in new_sums:
                        y_blocked = True
                if y_blocked:
                    score += 1
            
            if score > max_blocks:
                max_blocks = score
                best_x = x
                ties = []
            elif score == max_blocks:
                ties.append(x)
        
        if best_x is None:
            if candidates:
                best_x = rng.choice(candidates)
            else:
                break
        elif ties:
            best_x = rng.choice(ties)
        
        ok, selected, sums = _try_add(selected, sums, best_x)
        if not ok:
            break
    
    return selected, sums

def _local_search_reduce(selected, sums, rng, deadline):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    sums = set(sums)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(selected)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            
            new_sums = set()
            valid = True
            n = len(candidate)
            for a_idx in range(n):
                a = candidate[a_idx]
                for b in candidate[a_idx:]:
                    s = a + b
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if not valid:
                    break
            
            if not valid:
                continue
            
            if _is_maximal(candidate, new_sums):
                selected = candidate
                sums = new_sums
                improved = True
                break
    
    return selected, sums

def _swap_search(selected, sums, rng, deadline):
    """Try to swap elements: remove one, add another, to reduce size or find better configurations."""
    selected = list(selected)
    sums = set(sums)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(selected)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try removing element i
            remaining = selected[:i] + selected[i+1:]
            rem_sums = set()
            valid = True
            n = len(remaining)
            for a_idx in range(n):
                a = remaining[a_idx]
                for b in remaining[a_idx:]:
                    s = a + b
                    if s in rem_sums:
                        valid = False
                        break
                    rem_sums.add(s)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if remaining is maximal
            if _is_maximal(remaining, rem_sums):
                # We found a smaller maximal set!
                selected = remaining
                sums = rem_sums
                improved = True
                break
            
            # Try swapping: remove i, add some other element
            # Find unblocked elements for remaining
            unblocked = _get_unblocked(remaining, rem_sums)
            if not unblocked:
                continue
            
            # Try a few swaps
            sample_size = min(10, len(unblocked))
            candidates = rng.sample(unblocked, sample_size) if len(unblocked) > sample_size else list(unblocked)
            rng.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    break
                if x == selected[i]:
                    continue
                ok, new_sel, new_sums = _try_add(remaining, rem_sums, x)
                if ok and len(new_sel) == len(selected):
                    # Same size, but different configuration - might lead to better reductions
                    # Check if this new configuration allows further reduction
                    if _is_maximal(new_sel, new_sums):
                        # Try to reduce from this new state
                        reduced, reduced_sums = _local_search_reduce(new_sel, new_sums, rng, deadline)
                        if len(reduced) < len(selected):
                            selected = reduced
                            sums = reduced_sums
                            improved = True
                            break
                        elif len(reduced) == len(selected) and rng.random() < 0.1:
                            # Occasionally accept same-size improvements for diversity
                            selected = reduced
                            sums = reduced_sums
                            improved = True
                            break
                if improved:
                    break
        
        if improved:
            break
    
    return selected, sums

def _hit_set_search(selected, sums, rng, deadline):
    """
    Hitting set approach: For each missing element x, it must be blocked by at least one sum.
    x is blocked if 2x is in sums, or x+a is in sums for some a in selected.
    We want to find a subset of selected elements whose removal still leaves all missing elements blocked.
    This is formulated as: find minimum subset S' of S such that for all missing x, x is blocked by S'.
    """
    selected = list(selected)
    sums = set(sums)
    n = len(selected)
    
    if n < 3:
        return selected, sums
    
    # Precompute: for each missing element x, which elements of selected block it
    sel_set = set(selected)
    missing = [x for x in range(1, N + 1) if x not in sel_set]
    
    # For each missing x, find which selected elements are "responsible" for blocking it
    # x is blocked if: 2x in sums, or (x+a) in sums for some a in selected
    # The sums that block x are: 2x (if present), and x+a for each a where x+a in sums
    # We need to trace back which pairs produce these sums
    
    # For each sum s in sums, store which pairs produce it
    sum_to_pairs = {}
    for i in range(n):
        for j in range(i, n):
            s = selected[i] + selected[j]
            if s not in sum_to_pairs:
                sum_to_pairs[s] = []
            sum_to_pairs[s].append((i, j))
    
    # For each missing x, determine which selected elements must be present to block it
    # x is blocked by:
    #   - 2x if 2x in sums -> requires the pair (a,a) where 2a=2x, i.e., a=x, but x is missing
    #     Actually 2x in sums means there exists a,b in selected with a+b=2x
    #   - x+a in sums for some a in selected -> requires pair (b,c) with b+c=x+a
    
    # Simplified: for each missing x, find the set of selected element indices that are "essential"
    # for blocking x. If we remove an element, x might become unblocked.
    
    # For each missing x, find all sums that block it
    x_blockers = {}
    for x in missing:
        blocking_sums = []
        if 2 * x in sums:
            blocking_sums.append(2 * x)
        for a in selected:
            if x + a in sums:
                blocking_sums.append(x + a)
        
        # For each blocking sum, find which elements are involved
        essential_elements = set()
        for s in blocking_sums:
            if s in sum_to_pairs:
                for (i, j) in sum_to_pairs[s]:
                    essential_elements.add(i)
                    essential_elements.add(j)
        
        if essential_elements:
            x_blockers[x] = essential_elements
        else:
            # x is not properly blocked - this shouldn't happen if we're maximal
            x_blockers[x] = set(range(n))  # all elements are essential
    
    # Now we have: for each missing x, a set of element indices that must be present
    # We want to find the minimum subset of elements that hits all these sets
    # This is a set cover / hitting set problem
    
    # Greedy hitting set: repeatedly pick the element that covers the most uncovered x's
    # Start with all elements, try to remove them
    
    # Alternative: try removing each element and check if maximality is preserved
    # This is similar to _local_search_reduce but with the hitting set perspective
    
    best = selected
    best_sums = sums
    
    # Try single removals
    for i in range(n):
        if time.time() > deadline:
            break
        candidate = selected[:i] + selected[i+1:]
        if not candidate:
            continue
        
        new_sums = set()
        valid = True
        m = len(candidate)
        for a_idx in range(m):
            a = candidate[a_idx]
            for b in candidate[a_idx:]:
                s = a + b
                if s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if not valid:
                break
        
        if not valid:
            continue
        
        if _is_maximal(candidate, new_sums):
            if len(candidate) < len(best):
                best = candidate
                best_sums = new_sums
    
    # Try double removals if single didn't work well
    if len(best) == n and n > 4:
        for i in range(len(best)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(best)):
                if time.time() > deadline:
                    break
                candidate = best[:i] + best[i+1:j] + best[j+1:]
                if len(candidate) < 2:
                    continue
                
                new_sums = set()
                valid = True
                m = len(candidate)
                for a_idx in range(m):
                    a = candidate[a_idx]
                    for b in candidate[a_idx:]:
                        s = a + b
                        if s in new_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                if _is_maximal(candidate, new_sums):
                    if len(candidate) < len(best):
                        best = candidate
                        best_sums = new_sums
    
    return best, best_sums

def _construct():
    start_time = time.time()
    # Leave 2 seconds buffer for Python to exit
    deadline = start_time + 57.0
    
    best = None
    best_size = float('inf')
    
    seeds = [42, 123, 999, 2024, 314, 1618, 777, 555, 888, 444, 111, 222, 333, 500, 600, 700, 1234, 5678, 91011, 24680]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        
        # Try a few initial configurations per seed
        for attempt in range(8):
            if time.time() > deadline:
                break
            
            selected = []
            sums = set()
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
            # Start with a small random Sidon set
            target_size = rng.randint(3, 12)
            for x in candidates:
                if len(selected) >= target_size:
                    break
                ok, selected, sums = _try_add(selected, sums, x)
            
            if not selected:
                continue
            
            # Maximize
            selected, sums = _greedy_maximize(selected, sums, rng, deadline)
            
            # Validate
            if not _is_valid_sidon(selected):
                continue
            if not _is_maximal(selected, sums):
                continue
            
            # Update best
            if len(selected) < best_size:
                best_size = len(selected)
                best = list(selected)
            
            # Refine with local search
            if time.time() < deadline and best_size > 8:
                selected2, sums2 = _local_search_reduce(selected, sums, rng, deadline)
                if _is_valid_sidon(selected2) and _is_maximal(selected2, sums2):
                    if len(selected2) < best_size:
                        best_size = len(selected2)
                        best = list(selected2)
                        selected, sums = selected2, sums2
            
            # Refine with swap search
            if time.time() < deadline and best_size > 8:
                selected3, sums3 = _swap_search(selected, sums, rng, deadline)
                if _is_valid_sidon(selected3) and _is_maximal(selected3, sums3):
                    if len(selected3) < best_size:
                        best_size = len(selected3)
                        best = list(selected3)
                        selected, sums = selected3, sums3
            
            # Refine with hitting set search
            if time.time() < deadline and best_size > 8:
                selected4, sums4 = _hit_set_search(selected, sums, rng, deadline)
                if _is_valid_sidon(selected4) and _is_maximal(selected4, sums4):
                    if len(selected4) < best_size:
                        best_size = len(selected4)
                        best = list(selected4)
                        selected, sums = selected4, sums4
            
            # If we found a very small set, we can stop early
            if best_size <= 10:
                break
    
    if best is None:
        # Fallback: deterministic greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok, selected, sums = _try_add(selected, sums, x)
        return selected
    
    return best

def construct():
    return _construct()
