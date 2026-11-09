"""Combined program: vectorized validation + greedy + local search + min-conflicts."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

_N = 192
_MAX = 48
_DEADLINE = 85.0  # seconds, with 5s safety margin before 90s kill

def _check_validity(selected):
    """Check if the set is valid: no element divides sum of any 3 distinct others."""
    if any(x is None for x in selected):
        return False
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for a in selected:
        others = [v for v in selected if v != a]
        for triple in combinations(others, 3):
            if (triple[0] + triple[1] + triple[2]) % a == 0:
                return False
    return True

def _count_violations(selected):
    """Count total divisibility violations in the set."""
    if any(x is None for x in selected):
        return float('inf')
    n = len(selected)
    if n < 4:
        return 0
    count = 0
    for a in selected:
        others = [v for v in selected if v != a]
        for triple in combinations(others, 3):
            if (triple[0] + triple[1] + triple[2]) % a == 0:
                count += 1
    return count

def _can_add(x, selected):
    """Check if x can be added to selected without creating a violation."""
    n = len(selected)
    # Check 1: x divides sum of any 3 distinct elements in selected
    if n >= 3:
        for i in range(n):
            si = selected[i]
            for j in range(i + 1, n):
                sj = selected[j]
                s_ij = si + sj
                for k in range(j + 1, n):
                    if (s_ij + selected[k]) % x == 0:
                        return False
    
    # Check 2: For each a in selected, a | (x + b + c) for some b,c in selected\{a}
    for a in selected:
        for i in range(n):
            if selected[i] == a:
                continue
            bi = selected[i]
            for j in range(n):
                if j == i or selected[j] == a:
                    continue
                cj = selected[j]
                if (x + bi + cj) % a == 0:
                    return False
    return True

def _greedy_largest_first():
    """Greedy that picks largest elements first."""
    selected = []
    for x in range(_N, 0, -1):
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _greedy_smallest_first():
    """Greedy that picks smallest elements first."""
    selected = []
    for x in range(1, _N + 1):
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _greedy_random(seed):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    selected = []
    candidates = list(range(1, _N + 1))
    rng.shuffle(candidates)
    for x in candidates:
        if len(selected) >= _MAX:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit=15.0):
    """Enhanced local search: add, 1-for-1 swap, 2-for-2 swap."""
    if not _check_validity(selected):
        return selected
    
    start = time.time()
    best = list(selected)
    best_set = set(best)
    
    while time.time() - start < time_limit:
        improved = False
        candidates = [x for x in range(1, _N + 1) if x not in best_set]
        if not candidates:
            break
        
        # Try adding
        for x in candidates:
            if _can_add(x, best):
                best.append(x)
                best_set.add(x)
                improved = True
                break
        
        if not improved and len(best) >= 1:
            # Try 1-for-1 swap
            swap_done = False
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    break
                removed = best[i]
                temp_list = best[:i] + best[i + 1:]
                
                for x in candidates:
                    if x in best_set:
                        continue
                    if _can_add(x, temp_list):
                        best = temp_list + [x]
                        best_set = best_set - {removed} | {x}
                        swap_done = True
                        break
                
                if swap_done:
                    break
        
        if not improved and not swap_done:
            # Try 2-for-2 swap
            found = False
            if len(best) >= 2:
                for i in range(len(best)):
                    if time.time() - start > time_limit:
                        break
                    for j in range(i + 1, len(best)):
                        r1, r2 = best[i], best[j]
                        temp_list = [v for k, v in enumerate(best) if k != i and k != j]
                        
                        added_count = 0
                        for x in candidates:
                            if x in best_set:
                                continue
                            if _can_add(x, temp_list):
                                temp_list.append(x)
                                added_count += 1
                                if added_count == 2:
                                    break
                        
                        if added_count > 0:
                            best = temp_list
                            best_set = set(best)
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                break
    
    return best

def _min_conflicts_search(start_size=48, time_limit=30.0):
    """Min-conflicts local search: start with random 48-element subset, minimize violations."""
    start = time.time()
    rng = random.Random(42)
    
    # Start with random 48-element subset
    all_candidates = list(range(1, _N + 1))
    if len(all_candidates) < start_size:
        return all_candidates
    current = rng.sample(all_candidates, start_size)
    current_set = set(current)
    
    current_violations = _count_violations(current)
    best = list(current)
    best_violations = current_violations
    
    if current_violations == 0:
        return current
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start < time_limit and no_improve_count < max_no_improve:
        if current_violations == 0:
            return current
        
        # Pick a random index to swap
        idx = rng.randint(0, len(current) - 1)
        elem_to_remove = current[idx]
        
        # Try swapping with a random element from the unselected pool
        unselected = [x for x in range(1, _N + 1) if x not in current_set]
        if not unselected:
            break
            
        swap_with = rng.choice(unselected)
        
        # Perform swap
        current[idx] = swap_with
        current_set.discard(elem_to_remove)
        current_set.add(swap_with)
        
        new_violations = _count_violations(current)
        
        if new_violations < current_violations:
            current_violations = new_violations
            no_improve_count = 0
            if current_violations < best_violations:
                best_violations = current_violations
                best = list(current)
                if best_violations == 0:
                    return best
        else:
            # Revert swap
            current[idx] = elem_to_remove
            current_set.discard(swap_with)
            current_set.add(elem_to_remove)
            no_improve_count += 1
    
    return best if best_violations == 0 else current

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    
    # Start with multiple greedy strategies
    result = _greedy_largest_first()
    
    # Try smallest first
    alt = _greedy_smallest_first()
    if len(alt) > len(result):
        result = alt
    
    # Try random orderings
    for seed in range(50):
        if time.time() > deadline - 20:
            break
        alt = _greedy_random(seed)
        if len(alt) > len(result):
            result = alt
    
    # Local search to improve
    if time.time() < deadline - 10:
        result = _local_search(result, time_limit=10.0)
    
    # Min-conflicts search if we haven't reached max
    if len(result) < _MAX and time.time() < deadline - 5:
        mc_result = _min_conflicts_search(start_size=_MAX, time_limit=5.0)
        if len(mc_result) > len(result):
            result = mc_result
    
    return result
# EVOLVE-BLOCK-END
