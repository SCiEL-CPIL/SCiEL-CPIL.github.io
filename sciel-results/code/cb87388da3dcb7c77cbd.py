import time
import random
import itertools

def _count_comparable(family):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(family)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = family[i], family[j]
            if len(a) < len(b):
                if all(x in b for x in a):
                    count += 1
            elif len(b) < len(a):
                if all(x in a for x in b):
                    count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    all_ones = total_masks - 1
    
    # Precompute comparability bitsets for all 4096 masks
    # comp_bitset[i] is a 4096-bit integer where bit j is set iff i and j are comparable (i != j)
    comp_bitset = [0] * total_masks
    
    for i in range(total_masks):
        ci = 0
        # Enumerate all proper submasks of i
        if i != 0:
            sub = (i - 1) & i
            while True:
                ci |= (1 << sub)
                if sub == 0:
                    break
                sub = (sub - 1) & i
        # Enumerate all proper supermasks of i
        comp = all_ones ^ i
        if comp != 0:
            sub = (comp - 1) & comp
            while True:
                s = i | sub
                if s != i:
                    ci |= (1 << s)
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        comp_bitset[i] = ci
    
    # Greedy construction with good initial structure
    # Start with empty set (comparable with everything), then greedily add masks
    selected = []
    selected_mask = 0  # bitmask of selected mask indices
    
    # First, add the empty set (comparable with everything)
    selected.append(0)
    selected_mask |= (1 << 0)
    
    # Then greedily add masks that maximize comparability with current selection
    for _ in range(m - 1):
        best_mask = -1
        best_score = -1
        for k in range(total_masks):
            if (selected_mask >> k) & 1:
                continue
            score = (comp_bitset[k] & selected_mask).bit_count()
            if score > best_score:
                best_score = score
                best_mask = k
        selected.append(best_mask)
        selected_mask |= (1 << best_mask)
    
    # Calculate current score
    current_score = 0
    for i in range(m):
        current_score += (comp_bitset[selected[i]] & selected_mask).bit_count()
    current_score //= 2
    
    best_score = current_score
    best_selected = list(selected)
    
    # Local search: single swaps
    # Use bitset-based fast gain/loss computation
    def try_single_swap(selected, selected_mask, current_score, deadline):
        """Try to improve by swapping one set with another."""
        improved = False
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_mask = selected[idx]
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
            
            # Find best new mask to swap in
            best_new_mask = -1
            best_delta = 0
            for new_mask in range(total_masks):
                if (selected_mask >> new_mask) & 1:
                    continue
                gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_delta > 0 and best_new_mask != -1:
                # Apply the swap
                selected[idx] = best_new_mask
                selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new_mask)
                current_score += best_delta
                improved = True
                break  # restart scan from beginning
        
        return selected, selected_mask, current_score, improved
    
    # Iterative local search with multiple passes
    max_passes = 50
    for _ in range(max_passes):
        if time.time() >= deadline:
            break
        selected, selected_mask, current_score, improved = try_single_swap(
            selected, selected_mask, current_score, deadline
        )
        if current_score > best_score:
            best_score = current_score
            best_selected = list(selected)
        if not improved:
            break
    
    # If we have time left, try a few random restarts
    random.seed(42)
    restarts_done = 0
    max_restarts = 5
    
    while time.time() < deadline - 10.0 and restarts_done < max_restarts:
        restarts_done += 1
        
        # Random restart: pick m random masks
        restart_selected = []
        restart_mask = 0
        available = list(range(total_masks))
        random.shuffle(available)
        for k in range(m):
            restart_selected.append(available[k])
            restart_mask |= (1 << available[k])
        
        # Calculate restart score
        restart_score = 0
        for i in range(m):
            restart_score += (comp_bitset[restart_selected[i]] & restart_mask).bit_count()
        restart_score //= 2
        
        # Local search on the restart
        for _ in range(30):
            if time.time() >= deadline - 5.0:
                break
            restart_selected, restart_mask, restart_score, improved = try_single_swap(
                restart_selected, restart_mask, restart_score, deadline - 5.0
            )
            if not improved:
                break
        
        if restart_score > best_score:
            best_score = restart_score
            best_selected = list(restart_selected)
    
    # Convert to lists of integers
    result = []
    for mask in best_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure proper types
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
    
    return final_result
