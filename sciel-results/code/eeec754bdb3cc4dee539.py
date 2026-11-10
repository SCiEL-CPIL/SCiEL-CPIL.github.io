    def construct():
        deadline = time.time() + 85.0
        N = 256
        MAX_SIZE = 60
        
        # ... (keep check_valid, check_intervals) ...
        
        def dfs_search(target_size, current, start_idx):
            """
            Recursively try to build a set of target_size.
            current: list of chosen elements
            start_idx: index in pool to start picking from
            Returns: list of elements if successful, None otherwise.
            """
            if len(current) == target_size:
                return current
            
            # Pruning: if we don't have enough elements left
            if len(pool) - start_idx < target_size - len(current):
                return None
            
            # Try adding each candidate from start_idx onwards
            for i in range(start_idx, len(pool)):
                # Time check
                if time.time() > deadline - 2:
                    return None
                
                candidate = current + [pool[i]]
                
                # Fast check first?
                # If interval check passes, it's valid.
                # If interval check fails, we MUST use exact check.
                # So we can just use exact check always?
                # Exact check is fast enough for small N.
                
                if check_valid(candidate):
                    # Recurse
                    res = dfs_search(target_size, candidate, i + 1)
                    if res is not None:
                        return res
            
            return None
        
        pool = list(range(N, 0, -1)) # Descending order for better separation
        
        best = []
        
        # Try sizes from high to low? Or low to high to build up?
        # High to low finds the max faster if it exists.
        # But if max is 15, and we try 20, it might take forever to prove no solution.
        # Better: Start with a known good size (e.g. 8 from superincreasing) and try to increase.
        
        # Get initial best
        sup = [1]
        for i in range(1, 60):
            nxt = sum(sup) + 1
            if nxt <= N:
                sup.append(nxt)
            else:
                break
        if check_valid(sup):
            best = list(sup)
            
        # Try to improve
        current_best_size = len(best)
        
        # We can try to find a set of size current_best_size + 1
        # Then current_best_size + 2, etc.
        
        for target in range(current_best_size + 1, MAX_SIZE + 1):
            if time.time() > deadline - 5:
                break
            
            # To speed up DFS, we can use a heuristic order.
            # We already have pool sorted descending.
            
            found = dfs_search(target, [], 0)
            if found:
                best = found
                current_best_size = target
            else:
                # If we failed to find size T, it's likely we can't find T+1 either?
                # Not necessarily, but probably.
                # However, DFS might have failed due to time limit or bad path.
                # If it finished quickly, we can stop.
                # If it hit time limit, we should stop.
                break
                
        # Final check
        if not check_valid(best):
             # Fallback
             if check_valid(sup): best = sup
             else: best = [1]
             
        return [int(x) for x in best]
    