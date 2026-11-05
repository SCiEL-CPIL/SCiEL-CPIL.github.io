import time
import random
import math
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 52.0  # Safety margin before 90s kill

    # 1. Precompute all 3-APs
    three_aps = []
    for a in range(N):
        max_d = (N - 1 - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
    M = len(three_aps)
    
    # For fast delta calculation, we need to know for each element x,
    # which 3-APs it is part of, and what the other two elements are.
    # We will store this as a list of (other1, other2) for each x.
    # If x is the middle, other1=a, other2=c.
    # If x is the start, other1=b, other2=c.
    # If x is the end, other1=a, other2=b.
    # Actually, it's simpler to just check the specific pairs.
    
    # Let's precompute the pairs for each element x.
    # For each x, we want to know: 
    # 1. Pairs (y, z) such that {x, y, z} is a 3-AP.
    # If we add x, we gain a 3-AP if BOTH y and z are in the mask.
    
    neighbors = [[] for _ in range(N)]
    for a, b, c in three_aps:
        neighbors[a].append((b, c))
        neighbors[b].append((a, c))
        neighbors[c].append((a, b))

    # 2. Precompute 4-AP conflicts for validity check
    # For each x, list of tuples (y, z, w) such that {x,y,z,w} is a 4-AP.
    # We need to check if y, z, w are ALL in the mask.
    conflicts = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            conflicts[a].append((b, c, e))
            conflicts[b].append((a, c, e))
            conflicts[c].append((a, b, e))
            conflicts[e].append((a, b, c))

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (y, z, w) in conflicts[x]:
            if ((mask >> y) & 1) and ((mask >> z) & 1) and ((mask >> w) & 1):
                return False
        return True

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        for (y, z) in neighbors[x]:
            if ((mask >> y) & 1) and ((mask >> z) & 1):
                delta += 1
        return delta

    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask."""
        # Equivalent to delta_3aps_add on mask without x
        mask_without_x = mask & ~(1 << x)
        return delta_3aps_add(mask_without_x, x)

    def count_3aps(mask):
        """Count 3-APs in the current mask. Used for initialization and verification."""
        count = 0
        for a, b, c in three_aps:
            if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> c) & 1):
                count += 1
        return count

    # 3. Local Search Strategy
    # Start with a random valid set, then try to add elements that increase 3-AP count.
    # If stuck, try removing elements to allow others to be added? 
    # Actually, adding is usually the way to go. 
    # A better strategy: 
    # 1. Initialize with a random subset that is 4-AP free.
    # 2. While time remains:
    #    a. Try to add any element that is valid and increases 3-AP count.
    #    b. If no such element exists, try a "swap" or "remove" move to break local optima.
    #       Specifically, if we are stuck, we might have a "dense" set where no single addition helps,
    #       but removing a low-value element might allow two high-value elements to be added.
    
    best_mask = 0
    best_score = 0
    
    def greedy_init():
        """Create a random 4-AP-free set."""
        mask = 0
        elems = list(range(N))
        random.shuffle(elems)
        for x in elems:
            if is_valid_add(mask, x):
                # Accept with some probability to vary density
                if random.random() < 0.5:
                    mask |= (1 << x)
        return mask

    def local_search(mask):
        """Perform local search starting from mask. Returns improved mask and score."""
        current_mask = mask
        current_score = count_3aps(current_mask)
        
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Try all possible additions
            best_add = -1
            best_add_gain = 0
            for x in range(N):
                if not ((current_mask >> x) & 1):
                    if is_valid_add(current_mask, x):
                        gain = delta_3aps_add(current_mask, x)
                        if gain > best_add_gain:
                            best_add_gain = gain
                            best_add = x
            
            if best_add != -1:
                current_mask |= (1 << best_add)
                current_score += best_add_gain
                improved = True
                if current_score > best_score:
                    # We update global best here? No, let's return the local best.
                    pass
            else:
                # Stuck: No single addition improves the score.
                # Try a "remove" move to escape local optimum.
                # Find an element whose removal allows a net gain.
                # This is expensive. Instead, try random swaps or just restart.
                # Given the time limit, random restarts are often effective.
                # But let's try one step of "remove and re-add".
                
                # Identify elements with low contribution.
                # Contribution of x is delta_3aps_remove(current_mask, x).
                # If we remove x, we lose contrib(x).
                # Then we can try to add others.
                
                # Simple heuristic: Remove a random element with low contribution, 
                # then try to add best possible elements.
                
                # Let's find the element with the lowest contribution.
                min_contrib = float('inf')
                min_elem = -1
                for x in range(N):
                    if (current_mask >> x) & 1:
                        c = delta_3aps_remove(current_mask, x)
                        if c < min_contrib:
                            min_contrib = c
                            min_elem = x
                
                if min_elem != -1:
                    # Remove it
                    current_mask &= ~(1 << min_elem)
                    current_score -= min_contrib
                    
                    # Now try to add the best element
                    best_add = -1
                    best_add_gain = 0
                    for x in range(N):
                        if not ((current_mask >> x) & 1):
                            if is_valid_add(current_mask, x):
                                gain = delta_3aps_add(current_mask, x)
                                if gain > best_add_gain:
                                    best_add_gain = gain
                                    best_add = x
                                    
                    if best_add != -1:
                        current_mask |= (1 << best_add)
                        current_score += best_add_gain
                        improved = True # We made a move, even if score dropped or stayed same initially, we might climb later
                    else:
                        # Couldn't add anything back, score dropped.
                        # This is a bad move. Revert?
                        # For simplicity, if we can't improve, we accept the drop and continue searching?
                        # Or better: if the new score is worse than the score before the removal, revert.
                        # But we already updated current_score.
                        # Let's check if we should revert.
                        # If best_add_gain == 0, we lost min_contrib and gained 0. Net loss.
                        # If min_contrib was 0, net change 0.
                        # If min_contrib > 0, net loss.
                        # It's risky to accept net loss. 
                        # Let's just break and restart if we can't find a positive gain move.
                        break
                        
        return current_mask, current_score

    # Main loop: Multiple restarts
    num_restarts = 0
    while time.time() < deadline - 2.0:
        num_restarts += 1
        init_mask = greedy_init()
        local_mask, local_score = local_search(init_mask)
        
        if local_score > best_score:
            best_score = local_score
            best_mask = local_mask

    # Convert best mask to result
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)
