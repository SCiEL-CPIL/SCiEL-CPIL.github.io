import time
import random
import sys

def construct():
    """
    Find a maximum-size Sidon set (all pairwise sums a+b for a<=b distinct)
    within [1, 284] that must contain {1, 2, 5, 11}.
    
    Strategy:
    1. Greedy initialization to find a large valid set quickly.
    2. DFS with backtracking and pruning to try to find larger sets, 
       starting from the theoretical maximum (20) downwards.
    3. Use bytearray for fast sum conflict checking.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    prescribed = [1, 2, 5, 11]
    MAX_VAL = 284
    MAX_SUM = 2 * MAX_VAL
    SUMS_SIZE = MAX_SUM + 1
    
    # Helper to check if a candidate can be added to a set with given sums mask
    # sums_mask is a bytearray where sums_mask[s] = 1 if sum s is used
    def can_add(x, current_set, sums_mask):
        # Check 2*x
        if sums_mask[2 * x]:
            return False
        # Check x + y for all y in current_set
        for y in current_set:
            if sums_mask[x + y]:
                return False
        return True

    def add_to_set(x, current_set, sums_mask):
        sums_mask[2 * x] = 1
        for y in current_set:
            sums_mask[x + y] = 1
        current_set.append(x)

    def remove_from_set(x, current_set, sums_mask):
        current_set.pop()
        sums_mask[2 * x] = 0
        for y in current_set:
            sums_mask[x + y] = 0

    # 1. Greedy Initialization
    # We try a few different greedy orders to find a good initial solution
    best_set = []
    best_size = 0
    
    # Try descending order first (often better for spreading)
    # Then ascending
    orders = [list(range(MAX_VAL, 0, -1)), list(range(1, MAX_VAL + 1))]
    # Add a couple of random shuffles for diversity
    random.seed(42)
    for _ in range(2):
        shuffled = list(range(1, MAX_VAL + 1))
        random.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        if time.time() > deadline - 5:
            break
            
        current_set = list(prescribed)
        current_set.sort()
        sums_mask = bytearray(SUMS_SIZE)
        
        # Initialize sums for prescribed
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                sums_mask[current_set[i] + current_set[j]] = 1
                
        used = set(current_set)
        
        for x in order:
            if x in used:
                continue
            if can_add(x, current_set, sums_mask):
                add_to_set(x, current_set, sums_mask)
                used.add(x)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

    # Theoretical max is 20. We try to find a set of size 20, then 19, etc.
    # We use DFS starting from the prescribed elements.
    # To make DFS efficient, we pre-filter candidates that conflict with prescribed elements.
    
    # Pre-compute valid candidates relative to the prescribed set
    # A candidate x is valid if it doesn't conflict with any prescribed element
    prescribed_set = set(prescribed)
    base_sums = bytearray(SUMS_SIZE)
    for i in range(len(prescribed)):
        for j in range(i, len(prescribed)):
            base_sums[prescribed[i] + prescribed[j]] = 1
            
    valid_candidates = []
    for x in range(1, MAX_VAL + 1):
        if x in prescribed_set:
            continue
        if base_sums[2 * x]:
            continue
        conflict = False
        for y in prescribed:
            if base_sums[x + y]:
                conflict = True
                break
        if not conflict:
            valid_candidates.append(x)
            
    # Sort candidates in descending order for the DFS heuristic
    # This tends to find solutions with larger gaps earlier, which might be easier to extend
    # Or ascending? Let's stick with descending as it worked in the inspiration.
    valid_candidates.sort(reverse=True)
    
    def dfs(current_set, sums_mask, start_idx, target_size):
        global best_set, best_size
        
        if time.time() > deadline:
            return False
            
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = list(current_set)
            
        if current_size == target_size:
            return True
            
        # Pruning: Not enough candidates left
        if current_size + (len(valid_candidates) - start_idx) < target_size:
            return False
            
        for i in range(start_idx, len(valid_candidates)):
            x = valid_candidates[i]
            
            if can_add(x, current_set, sums_mask):
                add_to_set(x, current_set, sums_mask)
                
                if dfs(current_set, sums_mask, i + 1, target_size):
                    return True
                    
                remove_from_set(x, current_set, sums_mask)
                
        return False

    # Try to find a set of size 20, then 19, down to best_size + 1
    # We start from 20 because that is the known max for this range.
    for target in range(20, best_size, -1):
        if time.time() > deadline - 2:
            break
            
        # Reset state for this target
        current_set = list(prescribed)
        current_set.sort()
        sums_mask = bytearray(SUMS_SIZE)
        for i in range(len(current_set)):
            for j in range(i, len(current_set)):
                sums_mask[current_set[i] + current_set[j]] = 1
                
        if dfs(current_set, sums_mask, 0, target):
            # If we found a set of this size, it's optimal or near-optimal
            # We can return immediately if we found 20
            if target == 20:
                return best_set
            # If we found 19, we could try 20 again with different heuristics, 
            # but for simplicity and time safety, we take the first found max.
            # However, the loop goes down, so if we fail 20, we try 19.
            # If we succeed 19, we return it.
            return best_set

    return best_set

