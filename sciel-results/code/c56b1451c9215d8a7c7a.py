import random
import time
import math

def construct():
    N = 256
    
    # Precompute 3-APs and 4-APs
    three_aps = []
    four_aps = []
    
    # 3-APs: a, a+d, a+2d
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            three_aps.append(mask)
            
    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            four_aps.append(mask)
            
    # Precompute for each element, the masks of 3-APs and 4-APs it belongs to
    elem_to_threes = [[] for _ in range(N + 1)]
    elem_to_fours = [[] for _ in range(N + 1)]
    
    for mask in three_aps:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length()
            elem_to_threes[idx].append(mask)
            m ^= lsb
            
    for mask in four_aps:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length()
            elem_to_fours[idx].append(mask)
            m ^= lsb

    def get_score(mask):
        count = 0
        for ap in three_aps:
            if (mask & ap) == ap:
                count += 1
        return count

    def is_valid(mask):
        for ap in four_aps:
            if (mask & ap) == ap:
                return False
        return True

    def delta_add(mask, x):
        x_mask = 1 << (x - 1)
        gain = 0
        for ap_mask in elem_to_threes[x]:
            others = ap_mask & ~x_mask
            if (mask & others) == others:
                gain += 1
        return gain

    def delta_remove(mask, x):
        x_mask = 1 << (x - 1)
        loss = 0
        for ap_mask in elem_to_threes[x]:
            others = ap_mask & ~x_mask
            if (mask & others) == others:
                loss += 1
        return loss

    def can_add(mask, x):
        x_mask = 1 << (x - 1)
        for ap_mask in elem_to_fours[x]:
            others = ap_mask & ~x_mask
            if (mask & others) == others:
                return False
        return True

    def local_search_hill_climb(mask, score, time_limit):
        current_mask = mask
        current_score = score
        elements_in = [i for i in range(1, N+1) if current_mask & (1 << (i-1))]
        elements_out = [i for i in range(1, N+1) if not (current_mask & (1 << (i-1)))]
        
        start_time = time.time()
        improved = True
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            best_delta = 0
            best_move = None
            best_type = None # 'add', 'remove', 'swap'
            best_x = None
            best_y = None
            
            # Try Adds
            for y in elements_out:
                if can_add(current_mask, y):
                    gain = delta_add(current_mask, y)
                    if gain > best_delta:
                        best_delta = gain
                        best_move = 'add'
                        best_y = y
                        improved = True # We found a candidate, but we need to check all to find the BEST one? 
                        # For efficiency, we can just take the first improvement (First Ascent) or Best Ascent.
                        # Best Ascent is better for quality but slower.
                        # Let's do Best Ascent for small N.
                        
            # Try Removals (usually bad for score, but might be needed to enable swaps? 
            # No, in strict hill climb for max score, removal is never beneficial unless it allows a subsequent add 
            # that gains more. But that's a 2-step move. 
            # We can just try Swaps which cover this.)
            
            # Try Swaps
            # Optimized: Only try swaps if we have time or if we didn't find a good add.
            # Actually, let's just do a single pass of Best Ascent over all moves.
            
            # To speed up, we can randomize the order or sample if N is large. N=256 is small enough for full scan?
            # 256 adds + 256*256 swaps = 65k swaps. 
            # Each swap check: can_add (1000 ops) + delta_add (1000 ops) + delta_remove (1000 ops). 
            # 65k * 3000 = 195M ops. Too slow for one iteration in Python if done naively.
            
            # Optimization: 
            # 1. Pre-filter adds.
            # 2. For swaps, only consider 'promising' swaps?
            # Or use First Ascent: take the first improvement found.
            
            # Let's switch to First Ascent Hill Climbing for speed, with random restarts (ILS).
            
            # Restart logic:
            # 1. Randomly shuffle elements_out and elements_in.
            # 2. Iterate. If improvement found, apply, restart loop.
            
            # This is O(N) per step on average, much faster.
            
            pass # The previous block was pseudocode. Let's implement First Ascent ILS properly.

    # Implementation of ILS with First Ascent
    def ilds(time_limit):
        # Initial Solution: Random valid set
        mask = 0
        score = 0
        # Greedy random start
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(mask, x):
                mask |= (1 << (x-1))
                score += delta_add(mask ^ (1 << (x-1)), x) # Careful with delta calc order
        
        # Recalc score to be safe
        score = get_score(mask)
        
        best_mask = mask
        best_score = score
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Local Search: First Ascent Hill Climbing
            improved = True
            while improved:
                improved = False
                elements_in = [i for i in range(1, N+1) if mask & (1 << (i-1))]
                elements_out = [i for i in range(1, N+1) if not (mask & (1 << (i-1)))]
                
                random.shuffle(elements_in)
                random.shuffle(elements_out)
                
                # Try Adds
                for y in elements_out:
                    if can_add(mask, y):
                        gain = delta_add(mask, y)
                        if gain > 0:
                            mask |= (1 << (y-1))
                            score += gain
                            improved = True
                            break
                            
                if improved:
                    continue # Restart inner loop with new state
                
                # Try Swaps
                # To make it efficient, we don't check all N^2. We check a random subset or all if fast enough.
                # Let's check all swaps but break on first improvement.
                for x in elements_in:
                    # Optimization: If removing x loses 0 APs, it's a good candidate for swap
                    loss_x = delta_remove(mask, x)
                    if loss_x > 0:
                        continue # Removing x hurts score immediately. A swap would need gain_y > loss_x.
                        # We can still try, but let's prioritize loss_x == 0 first.
                        # For simplicity in first ascent, we just iterate.
                    
                    for y in elements_out:
                        # Check if y can be added to mask without x
                        temp_mask = mask & ~(1 << (x-1))
                        if can_add(temp_mask, y):
                            gain_y = delta_add(temp_mask, y)
                            delta = gain_y - loss_x
                            if delta > 0:
                                # Apply swap
                                mask = temp_mask | (1 << (y-1))
                                score += delta
                                improved = True
                                break
                    if improved:
                        break
                
            # Update Best
            if score > best_score:
                best_score = score
                best_mask = mask
                
            # Perturbation
            # Randomly remove 1-2 elements and add 1-2 valid elements
            # Or just do a random valid swap
            # Let's do: Randomly pick an element to remove (even if it hurts score) 
            # and then try to fill with adds/swaps in the next local search.
            # Actually, standard ILS perturbation:
            # Apply k random moves (can be bad).
            
            # Let's do: Randomly remove 1 element from current set.
            if mask:
                elements_in = [i for i in range(1, N+1) if mask & (1 << (i-1))]
                x_to_remove = random.choice(elements_in)
                mask &= ~(1 << (x_to_remove-1))
                score -= delta_remove(mask | (1 << (x_to_remove-1)), x_to_remove) # Recalc delta on old mask? 
                # Simpler: Recalc score
                score = get_score(mask)
                
            # Try to add some random valid elements to restore size?
            # The local search will do adds.
            
        return best_mask, best_score

    # Run ILS
    final_mask, final_score = ilds(55)
    
    result = []
    for i in range(1, N + 1):
        if final_mask & (1 << (i - 1)):
            result.append(i)
            
    return sorted(result)
