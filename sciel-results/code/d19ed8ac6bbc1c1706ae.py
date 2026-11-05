import time
from math import lcm

def construct():
    start_time = time.time()
    timeout = 55.0
    
    universe = list(range(1, 23))
    n_universe = len(universe)
    
    # Precompute global LCM and weights
    global_lcm = 1
    for x in universe:
        global_lcm = lcm(global_lcm, x)
        
    weights = {x: global_lcm // x for x in universe}

    # Order candidates: Small x first (Large weights)
    # This tends to create larger gaps, which helps avoid collisions
    ordered_universe = sorted(universe, key=lambda x: 1/x, reverse=True)
    # ordered_universe is [1, 2, 3, ..., 22]
    
    # Alternative order: Large x first?
    # Let's stick to small x first for now, as it mimics the binary expansion structure.
    
    def try_add(current_sums, w):
        """
        Check if adding weight w to current_sums causes a collision.
        Returns the new set of sums if valid, else None.
        """
        # Collision if any s in current_sums is equal to s' + w for some s' in current_sums
        # This is equivalent to checking if current_sums and (current_sums + w) intersect
        # But we need to be careful: we are adding w to ALL existing sums.
        # The new set is S' = S union (S + w).
        # Collision if S intersects (S + w).
        
        # Efficient check:
        # For each s in S, check if s - w is in S? No, that's for subtraction.
        # Check if s + w is in S?
        # If s + w is in S, then s and s-w (which is in S) both map to s?
        # Wait.
        # If $x \in S$ and $y \in S$ such that $x = y + w$, then $x$ is in $S$ and $y+w$ is in $S+w$.
        # So $x$ is in the intersection.
        
        # So we check: does any element in S differ from another element in S by w?
        # i.e., is there s1, s2 in S such that s1 - s2 = w?
        
        # Implementation:
        # Create a set of S.
        # For each s in S, check if s - w is in S.
        
        # Actually, simply:
        # new_sums = set(current_sums)
        # for s in current_sums:
        #     new_s = s + w
        #     if new_s in current_sums:
        #         return None
        #     new_sums.add(new_s)
        # return new_sums
        
        # The loop above is O(|S|).
        # We can optimize by checking intersection:
        # shifted = {s + w for s in current_sums}
        # if current_sums & shifted: return None
        # return current_sums | shifted
        
        # The set comprehension creates a new set, which is O(|S|).
        # The intersection is O(min(|S|, |shifted|)).
        # This is likely faster in Python due to C-level set operations.
        
        shifted = {s + w for s in current_sums}
        if current_sums & shifted:
            return None
        return current_sums | shifted

    def backtrack(current_set, current_sums, start_idx, target_size):
        if time.time() - start_time > timeout:
            return None
        
        if len(current_set) == target_size:
            return list(current_set)
        
        # Pruning: not enough elements left
        if len(current_set) + (n_universe - start_idx) < target_size:
            return None
            
        for i in range(start_idx, n_universe):
            x = ordered_universe[i]
            w = weights[x]
            
            new_sums = try_add(current_sums, w)
            if new_sums is not None:
                current_set.append(x)
                result = backtrack(current_set, new_sums, i + 1, target_size)
                if result is not None:
                    return result
                current_set.pop()
                
        return None

    best_set = []
    best_size = 0
    
    # Try to find maximum size
    # Upper bound: 22. But likely much smaller.
    # Let's start from 12 and go down.
    # If we find 12, we return it.
    # If we don't find 12 in time, we try 11, etc.
    # However, backtracking for 12 might be very slow if it's impossible.
    # We should keep track of the best set found during the search for any size.
    
    # To ensure we return *something* valid quickly, we can do a greedy pass first.
    # Greedy:
    greedy_set = []
    greedy_sums = {0}
    for x in ordered_universe:
        w = weights[x]
        new_sums = try_add(greedy_sums, w)
        if new_sums is not None:
            greedy_set.append(x)
            greedy_sums = new_sums
            
    best_set = greedy_set
    best_size = len(greedy_set)
    
    # Now try to improve with backtracking
    # Start from best_size + 1 up to 15? Or down from 15?
    # If we search for 15 and it fails, we wasted time.
    # Better: search for best_size + 1. If found, update best_size. Repeat.
    # This ensures we make progress.
    
    current_target = best_size + 1
    while current_target <= 15: # Cap at 15 for safety
        if time.time() - start_time > timeout - 5:
            break
            
        # We need to restart the search from scratch for the new target
        # But we can use the greedy set as a starting point?
        # No, backtracking needs to explore the space.
        # We can try to extend the greedy set?
        # If greedy found 10, can we find 11 by swapping?
        # Backtracking from scratch is safer but slower.
        
        # Optimization: If we found a set of size K, we can try to find K+1.
        # We can prune branches that don't contain elements from the known good set? No, that's too restrictive.
        
        result = backtrack([], {0}, 0, current_target)
        if result is not None:
            best_set = result
            best_size = current_target
            current_target += 1
        else:
            # If we couldn't find current_target, maybe a different ordering helps?
            # Or maybe current_target is the max.
            # Let's try one more time with a different ordering?
            # Or just break?
            # If backtracking for K fails, K+1 will also fail (usually).
            # So we can break.
            break

    return best_set
