"""Colorings with three colors in every four-term progression; optimized local search."""
import time
import random

# Precompute all nonconstant 4-term progressions within positions 1..128.
# A 4-AP is (a, a+d, a+2d, a+3d) with d>=1, a+3d<=128.
# Each AP is stored as a tuple of 4 0-based indices.
_ALL_APS = []
for d in range(1, 43):           # max d: 1+3d<=128 -> d<=42
    max_a = 128 - 3 * d
    for a in range(1, max_a + 1):
        _ALL_APS.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
_NUM_APS = len(_ALL_APS)

# For each position i, list of (ap_index, index_of_i_in_ap)
_POS_TO_APS = [[] for _ in range(128)]
for _ai, _ap in enumerate(_ALL_APS):
    for _idx_in_ap, _i in enumerate(_ap):
        _POS_TO_APS[_i].append((_ai, _idx_in_ap))

# Precompute "other positions" for each (ap_index, idx_in_ap)
# _AP_OTHERS[ai][j] = list of other 3 positions in AP ai
_AP_OTHERS = [[[] for _ in range(4)] for _ in range(_NUM_APS)]
for _ai, _ap in enumerate(_ALL_APS):
    for _j in range(4):
        _AP_OTHERS[_ai][_j] = [_ap[k] for k in range(4) if k != _j]


def _is_valid_modular(m):
    """Check if c_i = i % m is a valid coloring."""
    colors = [i % m for i in range(128)]
    for ap in _ALL_APS:
        c0 = colors[ap[0]]
        c1 = colors[ap[1]]
        c2 = colors[ap[2]]
        c3 = colors[ap[3]]
        distinct = len({c0, c1, c2, c3})
        if distinct < 3:
            return False
    return True


def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin under 90s kill

    # 1. Try simple modular constructions first (very fast)
    for m in range(3, 20):
        if time.time() > deadline - 1.0:
            break
        if _is_valid_modular(m):
            return [i % m for i in range(128)]

    # 2. Optimized Local Search with Bitmasks
    # We use bitmasks to track distinct colors in each AP.
    # This allows O(1) distinct count updates (via bit_count) and fast delta calculation.
    
    rng = random.Random(42)
    
    best_colors = None
    best_used = 999
    
    # Try increasing number of colors k
    for k in range(3, 8):
        if time.time() > deadline - 2.0:
            break
            
        found_valid = False
        max_trials = 5 if k == 3 else 10
        max_steps = 50000
        
        for trial in range(max_trials):
            if time.time() > deadline - 1.0:
                break
                
            # Initialize random coloring
            colors = [rng.randrange(k) for _ in range(128)]
            
            # Initialize masks and violation count
            # mask[ai] is an integer bitmask of colors present in AP ai
            masks = [0] * _NUM_APS
            total_viol = 0
            
            for ai, ap in enumerate(_ALL_APS):
                mask = 0
                for pos in ap:
                    mask |= (1 << colors[pos])
                masks[ai] = mask
                if mask.bit_count() < 3:
                    total_viol += 1
            
            # Local Search
            steps = 0
            stuck = 0
            while total_viol > 0 and steps < max_steps and time.time() < deadline - 1.0:
                steps += 1
                i = rng.randrange(128)
                old = colors[i]
                new = rng.randrange(k)
                if new == old:
                    continue
                
                # Calculate delta
                delta = 0
                # We need to update masks for all APs containing i
                # We do a "dry run" to see if we should accept
                # To avoid double work, we just compute the new masks for affected APs
                # and check if total_viol decreases.
                
                # Since the number of affected APs is small (~80), we can just simulate
                # the change on the masks directly, calculate new total_viol, and revert if bad.
                
                affected_aps = _POS_TO_APS[i]
                
                old_total_viol = total_viol
                new_total_viol = total_viol
                
                # Apply change to masks
                for ai, idx_in_ap in affected_aps:
                    old_mask = masks[ai]
                    new_mask = old_mask
                    
                    # Add new color
                    new_mask |= (1 << new)
                    
                    # Check if old color still exists in other positions
                    others = _AP_OTHERS[ai][idx_in_ap]
                    old_still_there = False
                    for o in others:
                        if colors[o] == old:
                            old_still_there = True
                            break
                    
                    if not old_still_there:
                        new_mask &= ~(1 << old)
                    
                    # Update violation status for this AP
                    old_distinct = old_mask.bit_count()
                    new_distinct = new_mask.bit_count()
                    
                    if old_distinct < 3 and new_distinct >= 3:
                        new_total_viol -= 1
                    elif old_distinct >= 3 and new_distinct < 3:
                        new_total_viol += 1
                    
                    masks[ai] = new_mask
                
                # Accept if better or equal (to allow mixing)
                if new_total_viol < old_total_viol:
                    total_viol = new_total_viol
                    colors[i] = new
                    stuck = 0
                elif new_total_viol == old_total_viol:
                    # Accept with some probability to escape plateaus
                    if rng.random() < 0.1:
                        total_viol = new_total_viol
                        colors[i] = new
                        stuck = 0
                    else:
                        # Revert masks
                        for ai, idx_in_ap in affected_aps:
                            old_mask = masks[ai]
                            new_mask = old_mask
                            new_mask |= (1 << new)
                            others = _AP_OTHERS[ai][idx_in_ap]
                            old_still_there = False
                            for o in others:
                                if colors[o] == old:
                                    old_still_there = True
                                    break
                            if not old_still_there:
                                new_mask &= ~(1 << old)
                            masks[ai] = new_mask
                        stuck += 1
                else:
                    # Revert masks
                    for ai, idx_in_ap in affected_aps:
                        old_mask = masks[ai]
                        new_mask = old_mask
                        new_mask |= (1 << new)
                        others = _AP_OTHERS[ai][idx_in_ap]
                        old_still_there = False
                        for o in others:
                            if colors[o] == old:
                                old_still_there = True
                                break
                        if not old_still_there:
                            new_mask &= ~(1 << old)
                        masks[ai] = new_mask
                    stuck += 1
                    
                    # Random restart if stuck too long
                    if stuck > 1000:
                        # Perturb
                        for _ in range(10):
                            j = rng.randrange(128)
                            colors[j] = rng.randrange(k)
                        # Recalculate masks and total_viol
                        total_viol = 0
                        for ai, ap in enumerate(_ALL_APS):
                            mask = 0
                            for pos in ap:
                                mask |= (1 << colors[pos])
                            masks[ai] = mask
                            if mask.bit_count() < 3:
                                total_viol += 1
                        stuck = 0
                        steps = 0 # reset steps to allow more search
                        
            if total_viol == 0:
                # Found a valid coloring with k colors
                used = len(set(colors))
                if used < best_used:
                    best_used = used
                    best_colors = colors[:]
                found_valid = True
                break # Stop trying more trials for this k, move to next k (smaller is better, so we break out of k loop if k is minimal? No, we iterate k upwards. If k=3 works, we return. If k=4 works, we check if k=3 worked earlier.)
        
        if found_valid:
            # We found a valid coloring with k colors.
            # Since we iterate k from 3 upwards, the first valid k is the minimum.
            # However, we might have found a valid coloring with k=4 in a previous iteration?
            # No, we break out of the inner loop, but not the outer k loop.
            # We should return immediately if we found a valid one with the current k,
            # because we are searching for the MINIMUM number of colors.
            # Wait, the loop is `for k in range(3, 8)`.
            # If k=3 fails, we go to k=4. If k=4 succeeds, we return k=4.
            # This is correct.
            return best_colors

    # If no valid coloring found with k<8, return the best attempt (likely invalid, score 0)
    # Or try a larger k with fewer steps?
    # Given the constraint, 3 or 4 colors should be sufficient for n=128.
    if best_colors:
        return best_colors
    else:
        # Fallback: return a random coloring (invalid)
        return [rng.randrange(3) for _ in range(128)]
