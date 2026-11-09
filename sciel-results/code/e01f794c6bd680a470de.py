"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
DEADLINE_MARGIN = 0.5  # seconds before the 30s limit to stop searching


# EVOLVE-BLOCK-START
def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set (all unordered sums a+b with a<=b distinct)."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _can_add(selected, sums, x):
    """Check if x can be added to selected without breaking Sidon property."""
    for a in selected:
        if (a + x) in sums:
            return False
    if (2 * x) in sums:
        return False
    return True


def _build_sums(selected):
    """Build the set of all unordered sums for a Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums


def _is_maximal(selected, sums):
    """Check if the Sidon set is inclusion-maximal in [1,N]."""
    sel_set = set(selected)
    for x in range(1, N + 1):
        if x not in sel_set and _can_add(selected, sums, x):
            return False
    return True


def _greedy_randomized(seed=None, deadline=None):
    """Greedy randomized construction."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    selected = []
    sums = set()
    
    for x in candidates:
        if _can_add(selected, sums, x):
            selected.append(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
        
        if deadline is not None and time.time() > deadline:
            break
    
    return selected


def _remove_element(selected, x):
    """Remove x from selected and rebuild sums. Returns (new_selected, new_sums) or None if invalid."""
    new_selected = [v for v in selected if v != x]
    if len(new_selected) != len(selected) - 1:
        return None
    new_sums = _build_sums(new_selected)
    return new_selected, new_sums


def _try_remove_and_repair(selected, sums, deadline=None):
    """
    Try to remove each element and see if the set remains maximal.
    If removing an element breaks maximality, try to add back the minimum
    number of elements to restore maximality.
    """
    sel_set = set(selected)
    best_size = len(selected)
    best_set = list(selected)
    
    for x in list(selected):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try removing x
        new_selected, new_sums = _remove_element(selected, x)
        if new_selected is None:
            continue
        
        # Check if still valid Sidon (it should be, since it's a subset)
        # Check if still maximal
        if _is_maximal(new_selected, new_sums):
            # Success! We found a smaller maximal set
            best_size = len(new_selected)
            best_set = new_selected
            # Continue trying to remove more
            selected = new_selected
            sums = new_sums
            sel_set = set(selected)
            # Restart the loop with the new set
            return _try_remove_and_repair(selected, sums, deadline)
        else:
            # Not maximal - try to repair by adding back minimal elements
            # Find which missing elements could be added
            missing = [y for y in range(1, N + 1) if y not in sel_set]
            # Try adding single elements to restore maximality
            for y in missing:
                if _can_add(new_selected, new_sums, y):
                    # Adding y might restore maximality
                    candidate = new_selected + [y]
                    cand_sums = set(new_sums)
                    for a in new_selected:
                        cand_sums.add(a + y)
                    cand_sums.add(2 * y)
                    
                    if len(candidate) < best_size and _is_maximal(candidate, cand_sums):
                        best_size = len(candidate)
                        best_set = candidate
                        selected = candidate
                        sums = cand_sums
                        sel_set = set(selected)
                        return _try_remove_and_repair(selected, sums, deadline)
    
    return best_set


def _simulated_annealing_remove(seed=None, deadline=None):
    """
    Simulated annealing approach: start with a valid maximal Sidon set,
    then try to remove elements. Accept removals that maintain maximality.
    Use random restarts for diversity.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Start with a greedy random set
    initial = _greedy_randomized(seed=seed, deadline=deadline)
    if not initial:
        return []
    
    sums = _build_sums(initial)
    
    # Verify initial is maximal
    if not _is_maximal(initial, sums):
        # Force maximality by adding elements
        for x in range(1, N + 1):
            if x not in set(initial) and _can_add(initial, sums, x):
                initial.append(x)
                for a in initial[:-1]:
                    sums.add(a + x)
                sums.add(2 * x)
    
    best = list(initial)
    best_size = len(initial)
    
    # Annealing: try random removals
    current = list(initial)
    current_sums = sums
    
    temperature = 1.0
    cooling_rate = 0.995
    
    while time.time() < deadline and temperature > 0.01:
        # Pick a random element to try removing
        if not current:
            break
        
        idx = rng.randint(0, len(current) - 1)
        x = current[idx]
        
        # Try removing x
        new_current = [v for i, v in enumerate(current) if i != idx]
        new_sums = _build_sums(new_current)
        
        # Check if still maximal
        if _is_maximal(new_current, new_sums):
            # Accept: it's a valid smaller maximal set
            current = new_current
            current_sums = new_sums
            if len(current) < best_size:
                best = list(current)
                best_size = len(current)
        else:
            # Not maximal - with some probability, accept anyway (to escape local optima)
            # but this would give an invalid result, so we don't accept non-maximal states
            # Instead, try a swap: remove x and add some other element
            sel_set = set(new_current)
            missing = [y for y in range(1, N + 1) if y not in sel_set]
            if missing:
                y = rng.choice(missing)
                if _can_add(new_current, new_sums, y):
                    candidate = new_current + [y]
                    cand_sums = set(new_sums)
                    for a in new_current:
                        cand_sums.add(a + y)
                    cand_sums.add(2 * y)
                    
                    if _is_maximal(candidate, cand_sums):
                        current = candidate
                        current_sums = cand_sums
                        if len(current) < best_size:
                            best = list(current)
                            best_size = len(current)
        
        temperature *= cooling_rate
    
    return best


def _iterative_deletion(seed=None, deadline=None):
    """
    Iterative deletion: repeatedly try to remove elements in a random order.
    This is a simpler version of simulated annealing.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Start with a greedy set
    current = _greedy_randomized(seed=seed, deadline=deadline)
    if not current:
        return []
    
    sums = _build_sums(current)
    
    # Ensure maximality
    for x in range(1, N + 1):
        if x not in set(current) and _can_add(current, sums, x):
            current.append(x)
            for a in current[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
    
    best = list(current)
    best_size = len(current)
    
    # Iteratively try to remove elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the order of removal attempts
        order = list(range(len(current)))
        rng.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            
            x = current[idx]
            new_current = [v for i, v in enumerate(current) if i != idx]
            new_sums = _build_sums(new_current)
            
            if _is_maximal(new_current, new_sums):
                current = new_current
                sums = new_sums
                improved = True
                if len(current) < best_size:
                    best = list(current)
                    best_size = len(current)
                # After a removal, restart the shuffle
                break
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 30 - DEADLINE_MARGIN
    
    best_set = None
    best_size = float('inf')
    
    # Strategy 1: Iterative deletion with multiple seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        s = _iterative_deletion(seed=seed, deadline=deadline)
        if s and len(s) < best_size:
            sums = _build_sums(s)
            if _is_valid_sidon(s) and _is_maximal(s, sums):
                best_set = s
                best_size = len(s)
    
    # Strategy 2: Simulated annealing removal with multiple seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        s = _simulated_annealing_remove(seed=seed, deadline=deadline)
        if s and len(s) < best_size:
            sums = _build_sums(s)
            if _is_valid_sidon(s) and _is_maximal(s, sums):
                best_set = s
                best_size = len(s)
    
    # Strategy 3: Random greedy as fallback
    for seed in range(20):
        if time.time() > deadline:
            break
        s = _greedy_randomized(seed=seed, deadline=deadline)
        if s and len(s) < best_size:
            sums = _build_sums(s)
            if _is_valid_sidon(s) and _is_maximal(s, sums):
                best_set = s
                best_size = len(s)
    
    # Final fallback: deterministic greedy
    if best_set is None:
        selected = []
        sums = set()
        for x in range(1, N + 1):
            if _can_add(selected, sums, x):
                selected.append(x)
                for a in selected[:-1]:
                    sums.add(a + x)
                sums.add(2 * x)
        best_set = selected
    
    # Final verification
    sums = _build_sums(best_set)
    assert _is_valid_sidon(best_set), "Result is not a valid Sidon set"
    assert _is_maximal(best_set, sums), "Result is not maximal"
    
    return best_set
# EVOLVE-BLOCK-END
