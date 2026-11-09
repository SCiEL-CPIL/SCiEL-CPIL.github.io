"""Rewritten program with optimized sunflower-free family construction."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = None
_DEADLINE = 90.0  # wall-clock seconds; evaluator kills at 90s, use safety margin

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

def _is_sunflower_triple(a, b, c):
    """Check if three distinct 4-sets form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _greedy_with_order(all_sets, order):
    """Greedy: iterate in given order, add set if it doesn't create a sunflower with any two chosen."""
    chosen = []
    chosen_masks = []
    for idx in order:
        s = all_sets[idx]
        ok = True
        # Check against all pairs in chosen
        for i in range(len(chosen_masks)):
            ai = chosen_masks[i]
            ai_s = ai & s
            for j in range(i + 1, len(chosen_masks)):
                aj = chosen_masks[j]
                ai_aj = ai & aj
                aj_s = aj & s
                if ai_s == ai_aj == aj_s:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(idx)
            chosen_masks.append(s)
    return chosen

def _greedy_with_order_optimized(all_sets, order):
    """Faster greedy using early termination and bit operations."""
    chosen_masks = []
    for idx in order:
        s = all_sets[idx]
        ok = True
        n = len(chosen_masks)
        for i in range(n):
            ai = chosen_masks[i]
            ai_s = ai & s
            for j in range(i + 1, n):
                aj = chosen_masks[j]
                if (ai & aj) == ai_s and (aj & s) == ai_s:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen_masks.append(s)
    return chosen_masks

def _local_improvement(all_sets, chosen_set):
    """Try to improve by removing one set and adding another (1-swap)."""
    chosen_list = list(chosen_set)
    chosen_set = set(chosen_list)
    n_all = len(all_sets)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        # Try removing each chosen set
        for remove_idx, rm_mask in enumerate(list(chosen_set)):
            trial_set = chosen_set - {rm_mask}
            # Try adding each non-chosen set
            for add_mask in all_sets:
                if add_mask in trial_set:
                    continue
                # Check if adding add_mask creates sunflower with any two in trial_set
                trial_list = list(trial_set)
                ok = True
                for i in range(len(trial_list)):
                    ai = trial_list[i]
                    ai_add = ai & add_mask
                    for j in range(i + 1, len(trial_list)):
                        aj = trial_list[j]
                        if (ai & aj) == ai_add and (aj & add_mask) == ai_add:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and len(trial_set) + 1 > len(chosen_set):
                    chosen_set = trial_set | {add_mask}
                    improved = True
                    break
            if improved:
                break
    return chosen_set

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    all_sets = _all_4sets_bitmask()
    n_all = len(all_sets)
    
    # Precompute for faster checking
    best_chosen = None
    best_size = 0
    
    # Try multiple random orderings
    rng = random.Random(42)
    max_rounds = 200  # will be limited by time
    
    for round_num in range(max_rounds):
        if _time_left() < 5.0:
            break
        
        # Create a random order with some structure
        order = list(range(n_all))
        rng.shuffle(order)
        
        chosen_masks = _greedy_with_order_optimized(all_sets, order)
        
        if len(chosen_masks) > best_size:
            best_size = len(chosen_masks)
            best_chosen = set(chosen_masks)
        
        # Local improvement
        if _time_left() > 10.0:
            improved = _local_improvement(all_sets, best_chosen)
            if len(improved) > best_size:
                best_size = len(improved)
                best_chosen = improved
    
    # Also try a deterministic lexicographic order for baseline
    if _time_left() > 5.0:
        lex_order = list(range(n_all))
        chosen_masks = _greedy_with_order_optimized(all_sets, lex_order)
        if len(chosen_masks) > best_size:
            best_size = len(chosen_masks)
            best_chosen = set(chosen_masks)
    
    # Convert back to sorted lists
    result = []
    for mask in best_chosen:
        elems = []
        for e in range(16):
            if mask & (1 << e):
                elems.append(e + 1)
        result.append(sorted(elems))
    
    return result
# EVOLVE-BLOCK-END
