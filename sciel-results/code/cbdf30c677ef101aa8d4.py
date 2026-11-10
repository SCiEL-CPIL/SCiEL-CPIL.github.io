import time
import random
import numpy as np
from collections import defaultdict

N = 256

# Precomputation
def _precompute():
    # 4-APs: (a, a+d, a+2d, a+3d) 0-based
    all_4aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # 3-APs: (a, a+d, a+2d) 0-based
    all_3aps = []
    for d in range(1, (N - 1) // 2 + 1):
        for a in range(0, N - 2 * d):
            all_3aps.append((a, a + d, a + 2 * d))
    
    # For each element, list of 4-AP indices it belongs to
    elem_to_4aps = [[] for _ in range(N)]
    for i, ap in enumerate(all_4aps):
        for x in ap:
            elem_to_4aps[x].append(i)
    
    # For each element, list of 3-AP indices it belongs to
    elem_to_3aps = [[] for _ in range(N)]
    for i, ap in enumerate(all_3aps):
        for x in ap:
            elem_to_3aps[x].append(i)
    
    # Masks for 4-APs and 3-APs
    masks_4ap = [(1 << a) | (1 << b) | (1 << c) | (1 << e) for a, b, c, e in all_4aps]
    masks_3ap = [(1 << a) | (1 << b) | (1 << c) for a, b, c in all_3aps]
    
    return all_4aps, all_3aps, elem_to_4aps, elem_to_3aps, masks_4ap, masks_3ap

(ALL_4APS, ALL_3APS, ELEM_TO_4APS, ELEM_TO_3APS, MASKS_4AP, MASKS_3AP) = _precompute()
N_4APS = len(ALL_4APS)
N_3APS = len(ALL_3APS)


def _count_3aps(mask):
    """Count 3-APs in mask."""
    count = 0
    for m in MASKS_3AP:
        if (mask & m) == m:
            count += 1
    return count


def _has_4ap_in_mask(mask, x):
    """Check if adding x to mask creates a 4-AP."""
    for ap_idx in ELEM_TO_4APS[x]:
        m = MASKS_4AP[ap_idx]
        # Check if the other 3 elements are in mask
        others = m & ~(1 << x)
        if (mask & others) == others:
            return True
    return False


def _count_3aps_delta_add(mask, x):
    """Count 3-APs created by adding x to mask."""
    count = 0
    for ap_idx in ELEM_TO_3APS[x]:
        m = MASKS_3AP[ap_idx]
        others = m & ~(1 << x)
        if (mask & others) == others:
            count += 1
    return count


def _count_3aps_delta_remove(mask, x):
    """Count 3-APs lost by removing x from mask (x must be in mask)."""
    count = 0
    for ap_idx in ELEM_TO_3APS[x]:
        m = MASKS_3AP[ap_idx]
        others = m & ~(1 << x)
        if (mask & others) == others:
            count += 1
    return count


def _greedy_add(rng, start_mask=0):
    """Greedy add: start from start_mask, add elements that maximize 3-AP delta."""
    mask = start_mask
    remaining = [x for x in range(N) if not (mask >> x) & 1]
    rng.shuffle(remaining)
    
    while remaining:
        best_x = -1
        best_delta = -1
        for x in remaining:
            if not _has_4ap_in_mask(mask, x):
                delta = _count_3aps_delta_add(mask, x)
                if delta > best_delta:
                    best_delta = delta
                    best_x = x
        if best_x == -1:
            break
        mask |= (1 << best_x)
        remaining.remove(best_x)
    
    return mask


def _greedy_remove(rng, start_mask=None):
    """Greedy remove: start from start_mask (or full), remove elements to eliminate 4-APs."""
    if start_mask is None:
        mask = (1 << N) - 1
    else:
        mask = start_mask
    
    # Track active 4-APs
    # We'll repeatedly find a 4-AP and remove the element that breaks the most 4-APs
    
    while True:
        # Find all active 4-APs
        active = []
        for i, m in enumerate(MASKS_4AP):
            if (mask & m) == m:
                active.append(i)
        
        if not active:
            break
        
        # Count involvement
        counts = [0] * N
        for ap_idx in active:
            for x in ALL_4APS[ap_idx]:
                counts[x] += 1
        
        max_count = max(counts)
        if max_count == 0:
            break
        candidates = [i for i in range(N) if counts[i] == max_count and (mask >> i) & 1]
        if not candidates:
            break
        x = rng.choice(candidates)
        mask &= ~(1 << x)
    
    return mask


def _local_search(mask, deadline, rng, tabu_tenure=15):
    """Local search with tabu list."""
    current_score = _count_3aps(mask)
    best_mask = mask
    best_score = current_score
    
    tabu = {}
    iter_count = 0
    no_improve = 0
    
    selected = [i for i in range(N) if (mask >> i) & 1]
    unselected = [i for i in range(N) if not ((mask >> i) & 1)]
    
    while time.time() < deadline and no_improve < 80:
        iter_count += 1
        moves = []
        
        # Add moves
        for x in unselected:
            if x in tabu and tabu[x] > iter_count:
                continue
            if not _has_4ap_in_mask(mask, x):
                delta = _count_3aps_delta_add(mask, x)
                moves.append(('add', x, delta))
        
        # Remove moves
        for x in selected:
            if x in tabu and tabu[x] > iter_count:
                continue
            delta = -_count_3aps_delta_remove(mask, x)
            moves.append(('remove', x, delta))
        
        # Swap moves (sampled)
        if selected and unselected:
            n_swap = min(len(selected), len(unselected), 25)
            sel_sample = rng.sample(selected, n_swap) if len(selected) > n_swap else selected
            unsel_sample = rng.sample(unselected, n_swap) if len(unselected) > n_swap else unselected
            
            for x_out in sel_sample:
                if x_out in tabu and tabu[x_out] > iter_count:
                    continue
                for x_in in unsel_sample:
                    if x_in in tabu and tabu[x_in] > iter_count:
                        continue
                    temp_mask = mask & ~(1 << x_out)
                    if not _has_4ap_in_mask(temp_mask, x_in):
                        delta_out = -_count_3aps_delta_remove(mask, x_out)
                        delta_in = _count_3aps_delta_add(temp_mask, x_in)
                        total_delta = delta_out + delta_in
                        moves.append(('swap', (x_out, x_in), total_delta))
        
        if not moves:
            break
        
        moves.sort(key=lambda m: m[2], reverse=True)
        best_move = moves[0]
        mtype, mdata, mdelta = best_move
        
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


def _ils_polish(mask, deadline, rng):
    """Iterated Local Search: perturb and re-localize."""
    best_mask = mask
    best_score = _count_3aps(mask)
    
    while time.time() < deadline:
        # Perturb
        perturbed = best_mask
        selected = [i for i in range(N) if (perturbed >> i) & 1]
        n_remove = rng.randint(3, 8)
        if len(selected) > n_remove:
            to_remove = rng.sample(selected, n_remove)
            for x in to_remove:
                perturbed &= ~(1 << x)
        
        # Local search
        ls_deadline = min(deadline, time.time() + 2.0)
        mask, score = _local_search(perturbed, ls_deadline, rng, tabu_tenure=12)
        
        if score > best_score:
            best_score = score
            best_mask = mask
    
    return best_mask, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safe margin
    
    rng = random.Random(42)
    
    best_mask = 0
    best_score = -1
    
    # Phase 1: Multiple diverse initializations
    strategies = []
    
    # Greedy add from empty
    strategies.append(('greedy_add', _greedy_add(rng)))
    
    # Greedy remove from full
    strategies.append(('greedy_remove', _greedy_remove(rng)))
    
    # Greedy add with random start (random 10 elements)
    for trial in range(3):
        rng2 = random.Random(100 + trial)
        start = 0
        for x in rng2.sample(range(N), 10):
            start |= (1 << x)
        strategies.append(('greedy_add_rand', _greedy_add(rng2, start)))
    
    # Greedy remove from random subset
    for trial in range(2):
        rng3 = random.Random(200 + trial)
        start = 0
        for x in rng3.sample(range(N), 128):
            start |= (1 << x)
        strategies.append(('greedy_remove_rand', _greedy_remove(rng3, start)))
    
    # Phase 2: Local search on each
    for name, mask in strategies:
        if time.time() > deadline - 10:
            break
        ls_deadline = min(deadline - 5, time.time() + 3.0)
        mask, score = _local_search(mask, ls_deadline, rng, tabu_tenure=15)
        if score > best_score:
            best_score = score
            best_mask = mask
    
    # Phase 3: ILS polish on best
    if time.time() < deadline - 5:
        best_mask, best_score = _ils_polish(best_mask, deadline - 2, rng)
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 1:
        best_mask, best_score = _local_search(best_mask, deadline, rng, tabu_tenure=20)
    
    # Convert to list
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    return sorted(result)
