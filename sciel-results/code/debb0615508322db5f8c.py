"""Rewritten program with optimized sunflower-free family construction."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 85.0  # wall-clock seconds; evaluator kills at 90s, use safety margin

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _all_4sets_bitmask():
    """Generate all 4-element subsets of [1,16] as bitmasks."""
    elems = list(range(16))
    result = []
    for combo in combinations(elems, 4):
        mask = 0
        for e in combo:
            mask |= (1 << e)
        result.append(mask)
    return result

def _popcount(x):
    return bin(x).count('1')

def _can_add(s, chosen_list):
    """Check if s can be added to chosen_list without forming a sunflower.
    For each pair (a,b) in chosen_list, check if (a&b) == (a&s) == (b&s)."""
    n = len(chosen_list)
    for i in range(n):
        ai = chosen_list[i]
        ai_s = ai & s
        for j in range(i + 1, n):
            aj = chosen_list[j]
            if (ai & aj) == ai_s and (aj & s) == ai_s:
                return False
    return True

def _greedy_fast(all_sets, order):
    """Greedy: iterate in given order, add set if it doesn't create a sunflower."""
    chosen_list = []
    chosen_set = set()
    for idx in order:
        s = all_sets[idx]
        if s in chosen_set:
            continue
        if _can_add(s, chosen_list):
            chosen_list.append(s)
            chosen_set.add(s)
    return chosen_set, chosen_list

def _local_improve_1swap(all_sets, chosen_set, chosen_list, max_candidates_per_rm=50, max_rm_attempts=50):
    """1-swap local improvement: try removing one set and adding another.
    Limit candidates tried per removal and number of removal attempts for speed."""
    improved = True
    rng = random.Random(123)
    while improved and _time_left() > 5.0:
        improved = False
        n_chosen = len(chosen_list)
        # Try a limited number of removals
        rm_indices = list(range(n_chosen))
        if len(rm_indices) > max_rm_attempts:
            rng.shuffle(rm_indices)
            rm_indices = rm_indices[:max_rm_attempts]
        for rm_idx in rm_indices:
            if _time_left() < 5.0:
                break
            rm_mask = chosen_list[rm_idx]
            # Build trial list without rm_mask
            trial_list = [chosen_list[i] for i in range(n_chosen) if i != rm_idx]
            trial_set = set(trial_list)
            # Try a limited number of non-chosen sets
            candidates = [s for s in all_sets if s not in trial_set]
            if len(candidates) > max_candidates_per_rm:
                rng.shuffle(candidates)
                candidates = candidates[:max_candidates_per_rm]
            for s in candidates:
                if _can_add(s, trial_list):
                    # Accept the swap
                    chosen_set = trial_set | {s}
                    chosen_list = trial_list + [s]
                    improved = True
                    break
            if improved:
                break
    return chosen_set, chosen_list

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    all_sets = _all_4sets_bitmask()
    n_all = len(all_sets)
    
    best_chosen_set = set()
    best_chosen_list = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Strategy 1: Multiple random greedy passes
    for round_num in range(500):
        if _time_left() < 10.0:
            break
        
        order = list(range(n_all))
        rng.shuffle(order)
        
        chosen_set, chosen_list = _greedy_fast(all_sets, order)
        
        if len(chosen_list) > best_size:
            best_size = len(chosen_list)
            best_chosen_set = set(chosen_list)
            best_chosen_list = list(chosen_list)
    
    # Strategy 2: Deterministic orders
    if _time_left() > 15.0:
        # Lexicographic
        order = list(range(n_all))
        chosen_set, chosen_list = _greedy_fast(all_sets, order)
        if len(chosen_list) > best_size:
            best_size = len(chosen_list)
            best_chosen_set = set(chosen_list)
            best_chosen_list = list(chosen_list)
        
        # Reverse lexicographic
        order = list(range(n_all - 1, -1, -1))
        chosen_set, chosen_list = _greedy_fast(all_sets, order)
        if len(chosen_list) > best_size:
            best_size = len(chosen_list)
            best_chosen_set = set(chosen_list)
            best_chosen_list = list(chosen_list)
    
    # Local improvement
    if _time_left() > 10.0:
        best_chosen_set, best_chosen_list = _local_improve_1swap(all_sets, best_chosen_set, best_chosen_list)
        if len(best_chosen_list) > best_size:
            best_size = len(best_chosen_list)
    
    # More local improvement rounds
    if _time_left() > 10.0:
        best_chosen_set, best_chosen_list = _local_improve_1swap(all_sets, best_chosen_set, best_chosen_list)
        if len(best_chosen_list) > best_size:
            best_size = len(best_chosen_list)
    
    # Convert back to sorted lists
    result = []
    for mask in best_chosen_set:
        elems = []
        for e in range(16):
            if mask & (1 << e):
                elems.append(e + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
