import time
import random
import numpy as np

N = 256

# Precomputation
# We use 0-based indexing for the bits in the integer mask.
# Element x in [1, N] corresponds to bit (x-1).

def _precompute():
    all_4aps = []
    all_3aps = []
    
    # Generate 4-APs: a, a+d, a+2d, a+3d
    # 0-based: a, a+d, a+2d, a+3d where 0 <= a < N, a+3d < N
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3*d):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            all_4aps.append((a, b, c, e))
            
    # Generate 3-APs: a, a+d, a+2d
    for d in range(1, (N - 1) // 2 + 1):
        for a in range(0, N - 2*d):
            b = a + d
            c = a + 2*d
            all_3aps.append((a, b, c))
            
    # Build other masks
    other_masks_4ap = [[] for _ in range(N)]
    for ap in all_4aps:
        a, b, c, e = ap
        # For a, others are b, c, e
        mask_a = (1 << b) | (1 << c) | (1 << e)
        other_masks_4ap[a].append(mask_a)
        # For b, others are a, c, e
        mask_b = (1 << a) | (1 << c) | (1 << e)
        other_masks_4ap[b].append(mask_b)
        # For c, others are a, b, e
        mask_c = (1 << a) | (1 << b) | (1 << e)
        other_masks_4ap[c].append(mask_c)
        # For e, others are a, b, c
        mask_e = (1 << a) | (1 << b) | (1 << c)
        other_masks_4ap[e].append(mask_e)
        
    other_masks_3ap = [[] for _ in range(N)]
    for ap in all_3aps:
        a, b, c = ap
        mask_a = (1 << b) | (1 << c)
        other_masks_3ap[a].append(mask_a)
        mask_b = (1 << a) | (1 << c)
        other_masks_3ap[b].append(mask_b)
        mask_c = (1 << a) | (1 << b)
        other_masks_3ap[c].append(mask_c)
        
    return all_4aps, other_masks_4ap, other_masks_3ap

ALL_4APS, OTHER_MASKS_4AP, OTHER_MASKS_3AP = _precompute()

def _has_4ap(mask, x):
    """Check if adding x (0-based) to mask creates a 4-AP."""
    # x is already in mask? No, we check if adding it creates one.
    # So we assume x is NOT in mask yet.
    for m in OTHER_MASKS_4AP[x]:
        if (mask & m) == m:
            return True
    return False

def _count_3aps_delta(mask, x):
    """Count 3-APs created by adding x (0-based)."""
    count = 0
    for m in OTHER_MASKS_3AP[x]:
        if (mask & m) == m:
            count += 1
    return count

def _count_3aps_total(mask):
    """Count total 3-APs in mask."""
    # We can sum deltas, but that's O(N^2).
    # Or iterate all 3-APs and check if all 3 bits are set.
    count = 0
    # all_3aps is not stored globally as masks, but we can generate masks on the fly or store them.
    # Let's just use the delta sum for a single element if we add one.
    # For total, we can do:
    # Sum of _count_3aps_delta for all x in mask / 3? No, each 3-AP is counted 3 times (once for each element).
    # So Total = (Sum_{x in mask} _count_3aps_delta(mask_without_x, x)) / 3.
    # This is expensive.
    # Better: Precompute masks for all 3-APs.
    pass

# Let's precompute 3-AP masks for total counting
ALL_3AP_MASKS = []
for d in range(1, (N - 1) // 2 + 1):
    for a in range(0, N - 2*d):
        b = a + d
        c = a + 2*d
        ALL_3AP_MASKS.append((1 << a) | (1 << b) | (1 << c))

def _count_3aps_total(mask):
    count = 0
    for m in ALL_3AP_MASKS:
        if (mask & m) == m:
            count += 1
    return count

def _greedy_add_construct(rng):
    mask = 0
    remaining = list(range(N))
    rng.shuffle(remaining) # Randomize order to break ties
    
    # We want to add elements that maximize delta.
    # To be efficient, we can just iterate through candidates.
    # Since N=256, we can just scan all remaining at each step.
    
    while remaining:
        best_x = -1
        best_delta = -1
        for x in remaining:
            if not _has_4ap(mask, x):
                delta = _count_3aps_delta(mask, x)
                if delta > best_delta:
                    best_delta = delta
                    best_x = x
        
        if best_x == -1:
            break
            
        mask |= (1 << best_x)
        remaining.remove(best_x)
        
    return mask

def _greedy_remove_construct(rng):
    mask = (1 << N) - 1 # All bits set
    # While there is a 4-AP, remove an element involved in the most 4-APs
    
    # Precompute list of 4-AP masks for fast checking?
    # We have ALL_4APS as tuples.
    # We can just iterate ALL_4APS to find active ones.
    
    while True:
        # Find all active 4-APs
        active_aps = []
        for ap in ALL_4APS:
            a, b, c, e = ap
            m = (1 << a) | (1 << b) | (1 << c) | (1 << e)
            if (mask & m) == m:
                active_aps.append(ap)
                
        if not active_aps:
            break
            
        # Count involvement of each element in active 4-APs
        counts = [0] * N
        for ap in active_aps:
            for x in ap:
                counts[x] += 1
                
        # Find element with max count
        max_count = max(counts)
        candidates = [i for i in range(N) if counts[i] == max_count]
        
        # Randomly pick one to break symmetry
        x = rng.choice(candidates)
        mask &= ~(1 << x)
        
    return mask

def _local_search(mask, deadline, rng):
    """Tabu Search / Local Search."""
    current_score = _count_3aps_total(mask)
    best_mask = mask
    best_score = current_score
    
    tabu = {}
    tabu_tenure = 10
    iter_count = 0
    no_improve = 0
    
    # Lists for fast access
    selected = [i for i in range(N) if (mask >> i) & 1]
    unselected = [i for i in range(N) if not ((mask >> i) & 1)]
    
    while time.time() < deadline and no_improve < 50:
        iter_count += 1
        moves = []
        
        # Add moves
        for x in unselected:
            if x in tabu and tabu[x] > iter_count:
                continue
            if not _has_4ap(mask, x):
                delta = _count_3aps_delta(mask, x)
                moves.append(('add', x, delta))
                
        # Remove moves
        for x in selected:
            if x in tabu and tabu[x] > iter_count:
                continue
            # Delta is negative of what we gain by keeping it?
            # If we remove x, we lose the 3-APs involving x.
            # delta = - _count_3aps_delta(mask, x)
            delta = - _count_3aps_delta(mask, x)
            moves.append(('remove', x, delta))
            
        # Swap moves (Sampled)
        if selected and unselected:
            n_swap = min(len(selected), len(unselected), 20)
            sel_sample = rng.sample(selected, n_swap) if len(selected) > n_swap else selected
            unsel_sample = rng.sample(unselected, n_swap) if len(unselected) > n_swap else unselected
            
            for x_out in sel_sample:
                if x_out in tabu and tabu[x_out] > iter_count:
                    continue
                for x_in in unsel_sample:
                    if x_in in tabu and tabu[x_in] > iter_count:
                        continue
                    # Check if swap is valid
                    # Remove x_out, add x_in
                    temp_mask = mask & ~(1 << x_out)
                    if not _has_4ap(temp_mask, x_in):
                        # Calculate delta
                        # Lose 3-APs with x_out (in original mask)
                        delta_out = - _count_3aps_delta(mask, x_out)
                        # Gain 3-APs with x_in (in temp_mask)
                        delta_in = _count_3aps_delta(temp_mask, x_in)
                        total_delta = delta_out + delta_in
                        moves.append(('swap', (x_out, x_in), total_delta))
                        
        if not moves:
            break
            
        moves.sort(key=lambda m: m[2], reverse=True)
        best_move = moves[0]
        mtype, mdata, mdelta = best_move
        
        # Apply move
        if mtype == 'add':
            x = mdata
            mask |= (1 << x)
            current_score += mdelta
            tabu[x] = iter_count + tabu_tenure
            unselected.remove(x)
            selected.append(x)
        elif mtype == 'remove':
            x = mdata
            mask &= ~(1 << x)
            current_score += mdelta
            tabu[x] = iter_count + tabu_tenure
            selected.remove(x)
            unselected.append(x)
        elif mtype == 'swap':
            x_out, x_in = mdata
            mask &= ~(1 << x_out)
            mask |= (1 << x_in)
            current_score += mdelta
            tabu[x_out] = iter_count + tabu_tenure
            tabu[x_in] = iter_count + tabu_tenure
            selected.remove(x_out)
            selected.append(x_in)
            unselected.remove(x_in)
            unselected.append(x_out)
            
        if current_score > best_score:
            best_score = current_score
            best_mask = mask
            no_improve = 0
        else:
            no_improve += 1
            
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85 # 5s margin
    
    rng = random.Random(42)
    
    best_mask = 0
    best_score = -1
    
    # Strategy: Mix of Greedy Add and Greedy Remove, followed by ILS
    
    n_restarts = 20
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Alternate construction strategies
        if restart % 2 == 0:
            mask = _greedy_add_construct(rng)
        else:
            mask = _greedy_remove_construct(rng)
            
        # Local Search
        ls_deadline = min(deadline - 5, time.time() + time_per_restart * 0.6)
        mask, score = _local_search(mask, ls_deadline, rng)
        
        # ILS Perturbation and Re-search
        if time.time() < deadline - 2:
            # Perturb: remove 5 random elements
            perturbed_mask = mask
            selected = [i for i in range(N) if (perturbed_mask >> i) & 1]
            if len(selected) > 10:
                to_remove = rng.sample(selected, 5)
                for x in to_remove:
                    perturbed_mask &= ~(1 << x)
                    
            # Re-localize
            ils_deadline = min(deadline - 2, time.time() + time_per_restart * 0.3)
            mask, score = _local_search(perturbed_mask, ils_deadline, rng)
            
        if score > best_score:
            best_score = score
            best_mask = mask
            
    # Convert to list
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    return sorted(result)
