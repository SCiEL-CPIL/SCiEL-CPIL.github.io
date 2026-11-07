# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n  # 4096
    all_ones = total_masks - 1  # 0xFFF
    
    # Precompute comparability bitsets
    # comp_bitset[i] has bit j set iff i and j are strictly comparable
    comp_bitset = [0] * total_masks
    
    for i in range(total_masks):
        ci = 0
        # Proper submasks of i
        sub = i
        while sub:
            ci |= (1 << sub)
            sub = (sub - 1) & i
        # Proper supermasks of i
        comp = all_ones ^ i
        if comp:
            sub = comp
            while sub:
                s = i | sub
                ci |= (1 << s)
                sub = (sub - 1) & comp
        comp_bitset[i] = ci
    
    def count_pairs(selected_mask, selected_list):
        """Count comparable pairs in the selected set."""
        total = 0
        for i in range(m):
            total += (comp_bitset[selected_list[i]] & selected_mask).bit_count()
        return total // 2
    
    def greedy_initial():
        """Greedy: start with empty set, then add masks maximizing comparability."""
        selected = []
        selected_mask = 0
        
        # Start with empty set (comparable with everything)
        selected.append(0)
        selected_mask |= 1
        
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
        
        return selected, selected_mask, count_pairs(selected_mask, selected)
    
    def chain_initial():
        """Initialize with a long chain, then fill greedily."""
        selected = []
        selected_mask = 0
        
        # Build a chain: empty, {0}, {0,1}, {0,1,2}, ... up to size 12
        mask = 0
        for i in range(n):
            selected.append(mask)
            selected_mask |= (1 << mask)
            mask |= (1 << i)
        # Now mask is all_ones, add it
        selected.append(mask)
        selected_mask |= (1 << mask)
        # That's 13 sets in the chain
        
        # Fill remaining greedily
        for _ in range(m - len(selected)):
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
        
        return selected, selected_mask, count_pairs(selected_mask, selected)
    
    def random_chain_initial():
        """Initialize with a random chain, then fill greedily."""
        selected = []
        selected_mask = 0
        
        # Build a random chain
        mask = 0
        remaining = list(range(n))
        random.shuffle(remaining)
        
        # Add empty set
        selected.append(mask)
        selected_mask |= (1 << mask)
        
        for i in range(n):
            mask |= (1 << remaining[i])
            selected.append(mask)
            selected_mask |= (1 << mask)
        
        # That's 13 sets in the chain
        
        # Fill remaining greedily
        for _ in range(m - len(selected)):
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
        
        return selected, selected_mask, count_pairs(selected_mask, selected)
    
    def local_search(selected, selected_mask, current_score, time_limit):
        """Exhaustive single-swap local search with best-improvement per position."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for idx in range(m):
                if time.time() >= time_limit:
                    break
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
                
                best_new = -1
                best_delta = 0
                for new_mask in range(total_masks):
                    if (selected_mask >> new_mask) & 1:
                        continue
                    gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new_mask
                
                if best_new >= 0:
                    selected[idx] = best_new
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << best_new)
                    current_score += best_delta
                    improved = True
        
        return selected, selected_mask, current_score
    
    def two_swap_local_search(selected, selected_mask, current_score, time_limit):
        """Try 2-swap moves: replace two elements with two new ones."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for idx1 in range(m - 1):
                if time.time() >= time_limit:
                    break
                for idx2 in range(idx1 + 1, m):
                    old1 = selected[idx1]
                    old2 = selected[idx2]
                    sel_without = selected_mask & ~(1 << old1) & ~(1 << old2)
                    loss = (comp_bitset[old1] & sel_without).bit_count() + \
                           (comp_bitset[old2] & sel_without).bit_count()
                    
                    # Try all pairs of new masks
                    best_new1 = -1
                    best_new2 = -1
                    best_delta = 0
                    
                    for new1 in range(total_masks):
                        if (selected_mask >> new1) & 1:
                            continue
                        if new1 == old1 or new1 == old2:
                            continue
                        gain1 = (comp_bitset[new1] & sel_without).bit_count()
                        for new2 in range(total_masks):
                            if (selected_mask >> new2) & 1:
                                continue
                            if new2 == old1 or new2 == old2 or new2 == new1:
                                continue
                            gain2 = (comp_bitset[new2] & sel_without).bit_count()
                            # Also count comparability between new1 and new2
                            cross_gain = 0
                            if (comp_bitset[new1] >> new2) & 1 or (comp_bitset[new2] >> new1) & 1:
                                cross_gain = 1
                            delta = gain1 + gain2 + cross_gain - loss
                            if delta > best_delta:
                                best_delta = delta
                                best_new1 = new1
                                best_new2 = new2
                    
                    if best_new1 >= 0:
                        selected[idx1] = best_new1
                        selected[idx2] = best_new2
                        selected_mask = (selected_mask & ~(1 << old1) & ~(1 << old2)) | (1 << best_new1) | (1 << best_new2)
                        current_score += best_delta
                        improved = True
        
        return selected, selected_mask, current_score
    
    # Try multiple initializations and keep the best
    best_score = -1
    best_selected = None
    best_mask = 0
    
    # Initial strategy 1: greedy from empty
    sel1, mask1, score1 = greedy_initial()
    if score1 > best_score:
        best_score = score1
        best_selected = list(sel1)
        best_mask = mask1
    
    # Initial strategy 2: chain-based
    sel2, mask2, score2 = chain_initial()
    if score2 > best_score:
        best_score = score2
        best_selected = list(sel2)
        best_mask = mask2
    
    # Initial strategy 3: random chain
    for _ in range(3):
        if time.time() > deadline - 10.0:
            break
        sel3, mask3, score3 = random_chain_initial()
        if score3 > best_score:
            best_score = score3
            best_selected = list(sel3)
            best_mask = mask3
    
    # Local search on the best initial
    sel, smask, score = local_search(list(best_selected), best_mask, best_score, deadline - 15.0)
    if score > best_score:
        best_score = score
        best_selected = list(sel)
        best_mask = smask
    
    # If we have time, try local search on other initials too
    if time.time() < deadline - 10.0:
        other_sel, other_mask, other_score = (sel2, mask2, score2) if best_mask == mask1 else (sel1, mask1, score1)
        sel4, smask4, score4 = local_search(list(other_sel), other_mask, other_score, deadline - 8.0)
        if score4 > best_score:
            best_score = score4
            best_selected = list(sel4)
            best_mask = smask4
    
    # Try 2-swap local search if we have time
    if time.time() < deadline - 5.0:
        sel5, smask5, score5 = two_swap_local_search(list(best_selected), best_mask, best_score, deadline - 3.0)
        if score5 > best_score:
            best_score = score5
            best_selected = list(sel5)
            best_mask = smask5
    
    # Final local search pass to polish
    if time.time() < deadline - 2.0:
        sel6, smask6, score6 = local_search(list(best_selected), best_mask, best_score, deadline - 1.0)
        if score6 > best_score:
            best_score = score6
            best_selected = list(sel6)
            best_mask = smask6
    
    # Convert to lists of integers
    result = []
    for mask in best_selected:
        result.append([int(b) for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
