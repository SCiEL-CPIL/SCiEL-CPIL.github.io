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
    """Fast Sidon check."""
    if len(selected) == 0:
        return True
    sums = _compute_sums_set(selected)
    total = len(selected) * (len(selected) + 1) // 2
    return len(sums) == total

def _is_maximal_fast(selected):
    """Check if selected is inclusion-maximal Sidon in [1,N]."""
    selected_set = set(selected)
    sums = _compute_sums_set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # x can be added iff 2x not in sums AND x+s not in sums for all s
        if 2 * x in sums:
            continue
        conflict = False
        for s in selected:
            if x + s in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _can_add(selected, sums, selected_set, x):
    """Check if x can be added to selected (Sidon-compatible)."""
    if x in selected_set:
        return False
    if 2 * x in sums:
        return False
    for s in selected:
        if x + s in sums:
            return False
    return True

def _greedy_construct(order, initial=None):
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
        if _can_add(selected, sums, selected_set, x):
            selected.append(x)
            new_sums = [x + a for a in selected[:-1]] + [2 * x]
            sums.update(new_sums)
            selected_set.add(x)
    return selected

def _forced_additions_v2(core, rng=None):
    """
    Given a Sidon core, greedily add elements to make it maximal.
    Pick the element that blocks the most 'addable' elements.
    """
    if rng is None:
        rng = random.Random()
    
    selected = list(core)
    sums = _compute_sums_set(selected)
    selected_set = set(selected)
    
    def _get_addable():
        addable = []
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            if _can_add(selected, sums, selected_set, x):
                addable.append(x)
        return addable
    
    max_iter = 500
    for _ in range(max_iter):
        addable = _get_addable()
        if not addable:
            break
        
        candidates = [x for x in range(1, N + 1) if x not in selected_set]
        best_candidate = None
        best_blocked = -1
        best_candidates = []
        
        for cand in candidates:
            if not _can_add(selected, sums, selected_set, cand):
                continue
            
            new_sums_list = [cand + a for a in selected] + [2 * cand]
            new_sums_set = sums | set(new_sums_list)
            new_selected = selected + [cand]
            new_selected_set = selected_set | {cand}
            
            blocked = 0
            for y in addable:
                if y == cand:
                    continue
                if not _can_add(new_selected, new_sums_set, new_selected_set, y):
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
    
    return selected

def _local_search_reduce(selected, deadline):
    """
    Try to remove elements while maintaining maximality.
    Uses swap moves to escape local minima.
    """
    current = list(selected)
    improved = True
    pass_count = 0
    max_passes = 30
    
    while improved and time.time() < deadline and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # Phase 1: Try removals
        for x in list(current):
            if time.time() > deadline:
                return current
            new_list = [y for y in current if y != x]
            if len(new_list) == 0:
                continue
            if _is_maximal_fast(new_list):
                current = new_list
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps
        current_set = set(current)
        for x in list(current):
            if time.time() > deadline:
                return current
            without_x = [y for y in current if y != x]
            if len(without_x) == 0:
                continue
            without_x_set = set(without_x)
            sums_x = _compute_sums_set(without_x)
            
            for y in range(1, N + 1):
                if y in without_x_set:
                    continue
                if not _can_add(without_x, sums_x, without_x_set, y):
                    continue
                
                candidate = without_x + [y]
                if _is_maximal_fast(candidate):
                    current = candidate
                    improved = True
                    break
            
            if improved:
                break
    
    return current

def _construct_spread_core(seed_val, target_size=8):
    """Construct a well-spread initial core."""
    rng = random.Random(seed_val)
    core = []
    
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if len(core) >= target_size:
            break
        if _can_add(core, _compute_sums_set(core), set(core), x):
            core.append(x)
    
    return core

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best = None
    best_size = N + 1
    
    # Strategy 1: Ascending greedy (baseline)
    order = list(range(1, N + 1))
    selected = _greedy_construct(order)
    if len(selected) < best_size:
        best = selected
        best_size = len(selected)
    
    if time.time() < deadline:
        reduced = _local_search_reduce(best, deadline)
        if len(reduced) < best_size:
            best = reduced
            best_size = len(reduced)
    
    # Strategy 2: Small cores with forced additions
    if time.time() < deadline - 5.0:
        for trial in range(500):
            if time.time() > deadline:
                break
            target_size = 5 + (trial % 6)  # Try cores of size 5-10
            core = _construct_spread_core(trial * 104729, target_size)
            if len(core) == 0:
                continue
            if not _is_sidon_fast(core):
                continue
            
            rng_comp = random.Random(trial * 31337)
            result = _forced_additions_v2(core, rng=rng_comp)
            
            if not _is_sidon_fast(result):
                continue
            if not _is_maximal_fast(result):
                continue
            
            if len(result) < best_size:
                best = result
                best_size = len(result)
                if time.time() < deadline - 2.0:
                    reduced = _local_search_reduce(best, deadline)
                    if len(reduced) < best_size:
                        best = reduced
                        best_size = len(reduced)
    
    # Strategy 3: Random orders with greedy
    if time.time() < deadline - 3.0:
        for trial in range(200):
            if time.time() > deadline:
                break
            rng = random.Random(trial * 7919 + 12345)
            order = list(range(1, N + 1))
            rng.shuffle(order)
            selected = _greedy_construct(order)
            if len(selected) < best_size:
                best = selected
                best_size = len(selected)
                if time.time() < deadline - 2.0:
                    reduced = _local_search_reduce(best, deadline)
                    if len(reduced) < best_size:
                        best = reduced
                        best_size = len(reduced)
    
    # Strategy 4: Different initial sizes
    if time.time() < deadline - 5.0:
        for init_size in range(3, 10):
            if time.time() > deadline:
                break
            for trial in range(30):
                if time.time() > deadline:
                    break
                rng = random.Random(trial * 7919 + init_size * 104729)
                initial = rng.sample(range(1, N + 1), min(init_size, N))
                initial.sort()
                
                if not _is_sidon_fast(initial):
                    continue
                
                remaining = [x for x in range(1, N + 1) if x not in set(initial)]
                rng.shuffle(remaining)
                
                selected = _greedy_construct(remaining, initial=initial)
                if len(selected) < best_size:
                    best = selected
                    best_size = len(selected)
                    if time.time() < deadline - 2.0:
                        reduced = _local_search_reduce(best, deadline)
                        if len(reduced) < best_size:
                            best = reduced
                            best_size = len(reduced)
    
    # Final safety check
    if best is None:
        best = _greedy_construct(list(range(1, N + 1)))
    
    return best
# EVOLVE-BLOCK-END
