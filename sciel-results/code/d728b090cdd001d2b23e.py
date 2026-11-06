"""Improved program: greedy with randomized restarts and deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-subsets of [16]
_ALL_SETS = [set(c) for c in combinations(range(1, 17), 4)]
_N_SETS = len(_ALL_SETS)

def _is_sunflower(s1, s2, s3):
    """Check if three sets form a sunflower: all pairwise intersections equal."""
    # Pairwise intersections must be equal
    inter12 = s1 & s2
    inter13 = s1 & s3
    inter23 = s2 & s3
    return inter12 == inter13 == inter23

def _can_add(candidate, chosen_sets):
    """Check if candidate can be added without forming a sunflower with any two existing sets."""
    for i in range(len(chosen_sets)):
        for j in range(i + 1, len(chosen_sets)):
            if _is_sunflower(candidate, chosen_sets[i], chosen_sets[j]):
                return False
    return True

def _greedy_with_order(order, deadline):
    """Greedy: process sets in given order, add if compatible."""
    chosen = []
    chosen_list = []  # list of set objects for fast intersection
    for idx in order:
        if time.time() > deadline:
            break
        s = _ALL_SETS[idx]
        # Check against all pairs in chosen
        ok = True
        for i in range(len(chosen_list)):
            if not ok:
                break
            for j in range(i + 1, len(chosen_list)):
                if _is_sunflower(s, chosen_list[i], chosen_list[j]):
                    ok = False
                    break
        if ok:
            chosen_list.append(s)
            chosen.append(idx)
    return chosen

def _greedy_with_order_fast(order, deadline):
    """Faster greedy using bitmask representation."""
    # Represent each 4-set as a 16-bit integer
    chosen_masks = []
    for idx in order:
        if time.time() > deadline:
            break
        mask = 0
        for el in _ALL_SETS[idx]:
            mask |= (1 << (el - 1))
        
        ok = True
        for i in range(len(chosen_masks)):
            if not ok:
                break
            for j in range(i + 1, len(chosen_masks)):
                # Check sunflower: pairwise intersections equal
                inter_ij = chosen_masks[i] & chosen_masks[j]
                inter_i_new = chosen_masks[i] & mask
                inter_j_new = chosen_masks[j] & mask
                if inter_ij != inter_i_new or inter_ij != inter_j_new:
                    ok = False
                    break
        if ok:
            chosen_masks.append(mask)
    return len(chosen_masks), chosen_masks

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin (evaluator kills at 90s)
    
    best_size = 0
    best_masks = []
    
    # Precompute bitmask for each set
    masks = []
    for s in _ALL_SETS:
        m = 0
        for el in s:
            m |= (1 << (el - 1))
        masks.append(m)
    
    # Multiple randomized greedy passes
    rng = random.Random(42)
    
    num_restarts = 500  # will be limited by deadline
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random permutation of indices
        order = list(range(_N_SETS))
        rng.shuffle(order)
        
        # Greedy with this order
        chosen_masks = []
        for idx in order:
            if time.time() > deadline - 2.0:
                break
            mask = masks[idx]
            
            ok = True
            for i in range(len(chosen_masks)):
                if not ok:
                    break
                for j in range(i + 1, len(chosen_masks)):
                    inter_ij = chosen_masks[i] & chosen_masks[j]
                    inter_i_new = chosen_masks[i] & mask
                    inter_j_new = chosen_masks[j] & mask
                    if inter_ij != inter_i_new or inter_ij != inter_j_new:
                        ok = False
                        break
            if ok:
                chosen_masks.append(mask)
        
        if len(chosen_masks) > best_size:
            best_size = len(chosen_masks)
            best_masks = list(chosen_masks)
            
            # Early exit if we hit the maximum possible
            if best_size >= 250:
                break
    
    # Convert best_masks back to sorted lists of elements
    result = []
    for m in best_masks:
        elements = [i + 1 for i in range(16) if (m >> i) & 1]
        result.append(sorted(elements))
    
    # If we got nothing, fallback to simple greedy
    if not result:
        result = _simple_greedy(deadline)
    
    return result

def _simple_greedy(deadline):
    """Simple deterministic greedy as fallback."""
    chosen = []
    for idx in range(_N_SETS):
        if time.time() > deadline:
            break
        s = _ALL_SETS[idx]
        ok = True
        for i in range(len(chosen)):
            if not ok:
                break
            for j in range(i + 1, len(chosen)):
                if _is_sunflower(s, chosen[i], chosen[j]):
                    ok = False
                    break
        if ok:
            chosen.append(s)
    return [sorted(s) for s in chosen]
# EVOLVE-BLOCK-END
