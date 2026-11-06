import time
import sys
import random
from collections import Counter

# Global variables
best_set = []
best_size = 0
start_time = 0
time_limit = 115.0

SUMS_SIZE = 570

def construct():
    global best_set, best_size, start_time
    
    start_time = time.time()
    
    initial_set = [1, 2, 5, 11]
    initial_sums = bytearray(SUMS_SIZE)
    
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            s = initial_set[i] + initial_set[j]
            initial_sums[s] = 1
            
    # 1. Greedy Construction to find a good initial lower bound
    # Strategy: Add the largest valid candidate repeatedly.
    # This spreads marks out, which is good for Sidon sets.
    
    all_candidates = list(range(284, 11, -1)) # Descending order
    
    greedy_set = initial_set[:]
    greedy_sums = initial_sums[:]
    
    for x in all_candidates:
        if 2 * x < SUMS_SIZE and greedy_sums[2 * x]:
            continue
        valid = True
        for y in greedy_set:
            s = x + y
            if s < SUMS_SIZE and greedy_sums[s]:
                valid = False
                break
        if valid:
            greedy_set.append(x)
            greedy_sums[2 * x] = 1
            for y in greedy_set:
                if y != x:
                    greedy_sums[x + y] = 1
                    
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set[:]
        
    # 2. Local Search: try to improve by swapping elements
    def local_search(start_set, max_iters=5000):
        """Try to improve the set by swapping elements or adding/removing."""
        current = start_set[:]
        current.sort()
        sums = bytearray(SUMS_SIZE)
        for i in range(len(current)):
            for j in range(i, len(current)):
                sums[current[i] + current[j]] = 1
        
        best_local = current[:]
        best_local_size = len(current)
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 2:
                break
                
            improved = False
            
            # Try adding a random element
            available = [x for x in range(12, 285) if x not in current]
            if available:
                x = random.choice(available)
                if 2 * x < SUMS_SIZE and not sums[2 * x]:
                    valid = True
                    for y in current:
                        s = x + y
                        if s < SUMS_SIZE and sums[s]:
                            valid = False
                            break
                    if valid:
                        current.append(x)
                        current.sort()
                        sums[2 * x] = 1
                        for y in current:
                            if y != x:
                                sums[x + y] = 1
                        improved = True
                        if len(current) > best_local_size:
                            best_local_size = len(current)
                            best_local = current[:]
                        break
            
            # Try swapping an element
            if len(current) >= 5:
                idx = random.randint(0, len(current) - 1)
                old = current[idx]
                available = [x for x in range(12, 285) if x not in current]
                if available:
                    x = random.choice(available)
                    # Check if x can replace old
                    temp_set = current[:]
                    temp_set.remove(old)
                    
                    valid = True
                    if 2 * x < SUMS_SIZE and sums[2 * x]:
                        valid = False
                    else:
                        for y in temp_set:
                            s = x + y
                            if s < SUMS_SIZE and sums[s] and s != old + y:
                                valid = False
                                break
                    
                    if valid:
                        # Perform swap
                        current[idx] = x
                        current.sort()
                        
                        # Update sums
                        sums[2 * old] = 0
                        for y in current:
                            if y != old:
                                sums[old + y] = 0
                        
                        sums[2 * x] = 1
                        for y in current:
                            if y != x:
                                sums[x + y] = 1
                        
                        improved = True
                        break
            
            if not improved:
                # Try removing an element to make room
                if len(current) > 5:
                    idx = random.randint(0, len(current) - 1)
                    old = current[idx]
                    current.remove(old)
                    sums[2 * old] = 0
                    for y in current:
                        sums[old + y] = 0
                    improved = True
        
        return best_local

    # Run local search from greedy result
    improved_set = local_search(greedy_set, max_iters=3000)
    if len(improved_set) > best_size:
        best_size = len(improved_set)
        best_set = improved_set[:]
        
    # 3. Improved DFS with better pruning
    candidates = list(range(12, 285))
    candidates.sort(reverse=True)
    
    K = len(candidates)
    
    # Precompute conflicts between candidates
    # conflict[i][j] = True if candidates[i] and candidates[j] cannot both be in the set
    # This is expensive to precompute fully, so we'll compute on the fly
    
    def can_add(x, current_set, current_sums):
        """Check if x can be added to current_set."""
        if 2 * x < SUMS_SIZE and current_sums[2 * x]:
            return False
        for y in current_set:
            s = x + y
            if s < SUMS_SIZE and current_sums[s]:
                return False
        return True
    
    def dfs(current_set, current_sums, avail_mask, target_size, depth):
        global best_set, best_size
        
        if time.time() - start_time > time_limit - 3:
            return False
            
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if current_size == target_size:
            return True
            
        # Pruning: Upper bound
        remaining = avail_mask.bit_count()
        if current_size + remaining < target_size:
            return False
            
        # Additional pruning: if we're deep and still far from target, bail out
        if depth > 10 and current_size + remaining < target_size + 2:
            return False
            
        # Iterate through available candidates
        mask = avail_mask
        while mask:
            lsb = mask & -mask
            i = lsb.bit_length() - 1
            mask ^= lsb
            
            x = candidates[i]
            
            if not can_add(x, current_set, current_sums):
                continue
                
            # Add x
            current_set.append(x)
            current_sums[2 * x] = 1
            for y in current_set:
                if y != x:
                    current_sums[x + y] = 1
            
            # Compute new_avail_mask by filtering conflicts
            new_mask = avail_mask & ~lsb
            
            # Filter out candidates that conflict with current_set
            temp_mask = new_mask
            filtered_mask = 0
            while temp_mask:
                lsb_j = temp_mask & -temp_mask
                j = lsb_j.bit_length() - 1
                temp_mask ^= lsb_j
                
                v = candidates[j]
                
                if can_add(v, current_set, current_sums):
                    filtered_mask |= lsb_j
            
            if dfs(current_set, current_sums, filtered_mask, target_size, depth + 1):
                return True
                
            # Backtrack
            current_set.pop()
            current_sums[2 * x] = 0
            for y in current_set:
                current_sums[x + y] = 0
                
        return False

    # Try to improve from best_size
    for target in range(best_size + 1, 21):
        if time.time() - start_time > time_limit - 8:
            break
            
        init_set_copy = initial_set[:]
        init_sums_copy = initial_sums[:]
        
        init_mask = (1 << K) - 1
        
        if dfs(init_set_copy, init_sums_copy, init_mask, target, 0):
            continue
        else:
            break
            
    return best_set
