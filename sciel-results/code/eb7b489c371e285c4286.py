"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _to_mask(s):
    m = 0
    for x in s:
        m |= (1 << (x - 1))
    return m

def _mask_to_set(m):
    return [i + 1 for i in range(16) if m & (1 << i)]

def _forms_sunflower(a, b, c):
    # Check if three sets form a sunflower: all three pairwise intersections are equal as sets
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _greedy_with_order(all_sets, order, deadline):
    chosen = []
    chosen_masks = []
    for idx in order:
        if time.time() >= deadline:
            break
        s = all_sets[idx]
        m = _to_mask(s)
        ok = True
        for i in range(len(chosen)):
            if time.time() >= deadline:
                break
            for j in range(i + 1, len(chosen)):
                if _forms_sunflower(chosen[i], chosen[j], s):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(s)
            chosen_masks.append(m)
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    best = []
    
    # First pass: original order
    result = _greedy_with_order(all_sets, list(range(n_sets)), deadline)
    if len(result) > len(best):
        best = result
    
    # Multiple randomized restarts
    for restart in range(200):
        if time.time() >= deadline:
            break
        order = list(range(n_sets))
        random.shuffle(order)
        result = _greedy_with_order(all_sets, order, deadline)
        if len(result) > len(best):
            best = result
    
    return [sorted(s) for s in best]
# EVOLVE-BLOCK-END
