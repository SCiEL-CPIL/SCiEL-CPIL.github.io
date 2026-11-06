import time
import random

def _find_subset(S, target):
    """
    Finds a subset of S that sums to target.
    Returns the subset as a set, or None if no such subset exists.
    """
    # Sort S to potentially find solutions faster or more consistently
    # Actually, order doesn't matter for existence, but for reconstruction path it does.
    # We just need ONE subset.
    
    # DP approach
    # reachable[s] is True if sum s is reachable
    # parent[s] is the previous sum
    # item_at[s] is the item added to reach s
    
    reachable = [False] * (target + 1)
    parent = [-1] * (target + 1)
    item_at = [-1] * (target + 1)
    
    reachable[0] = True
    
    for x in S:
        if x > target:
            continue
        # Iterate backwards to avoid using x multiple times in one step
        for s in range(target, x - 1, -1):
            if reachable[s - x] and not reachable[s]:
                reachable[s] = True
                parent[s] = s - x
                item_at[s] = x
                if s == target:
                    # Reconstruct the subset
                    subset = set()
                    curr = target
                    while curr != 0:
                        subset.add(item_at[curr])
                        curr = parent[curr]
                    return subset
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = None
    best_size = 0
    
    # Strategies for selecting which element to remove from the conflicting subset
    # 'min': remove smallest element (likely in many other subsets)
    # 'max': remove largest element (greedy large-first style)
    # 'rand': random choice
    strategies = ['min', 'max', 'rand']
    
    run_count = 0
    
    while time.time() < deadline:
        run_count += 1
        S = set(range(1, 101))
        
        # Select strategy
        if run_count % 3 == 1:
            strategy = 'min'
        elif run_count % 3 == 2:
            strategy = 'max'
        else:
            strategy = 'rand'
        
        # Lazy Hitting Set Loop
        while True:
            if time.time() > deadline:
                break
            
            F = _find_subset(S, 200)
            
            if F is None:
                # S is valid
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S.copy()
                break
            
            # Select element to remove
            if strategy == 'min':
                v = min(F)
            elif strategy == 'max':
                v = max(F)
            else:
                v = random.choice(list(F))
            
            S.remove(v)
        
        # If time is running out, stop generating new candidates
        if time.time() > deadline - 10.0:
            break

    # If we didn't find any valid set (should not happen, e.g., {100} is valid), 
    # fall back to a trivial valid set
    if best_set is None:
        best_set = {100}
        best_size = 1

    # Local Search: Try to add back removed elements
    if best_set is not None and time.time() < deadline - 5.0:
        S = best_set
        R = set(range(1, 101)) - S
        
        improved = True
        # Multiple passes to ensure we don't miss improvements due to order
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Shuffle R to avoid bias in addition order
            removals_list = list(R)
            random.shuffle(removals_list)
            
            for v in removals_list:
                if time.time() > deadline - 2.0:
                    break
                
                S_candidate = S.union({v})
                # Check if adding v creates a subset summing to 200
                # We can optimize this check: 
                # Does S contain a subset summing to 200 - v?
                # If yes, adding v creates a sum of 200.
                # So we just need to check if 200 - v is reachable in S.
                
                # Helper to check reachability of a specific target in a set
                def is_reachable(target_set, target_val):
                    if target_val < 0:
                        return False
                    if target_val == 0:
                        return True
                    reachable = [False] * (target_val + 1)
                    reachable[0] = True
                    for x in target_set:
                        if x > target_val:
                            continue
                        for s in range(target_val, x - 1, -1):
                            if reachable[s - x] and not reachable[s]:
                                reachable[s] = True
                                if s == target_val:
                                    return True
                    return reachable[target_val]

                if not is_reachable(S, 200 - v):
                    # Adding v is safe
                    S.add(v)
                    R.remove(v)
                    improved = True

        # Update best if local search improved it
        if len(S) > best_size:
            best_set = S
            best_size = len(S)

    return list(best_set)
