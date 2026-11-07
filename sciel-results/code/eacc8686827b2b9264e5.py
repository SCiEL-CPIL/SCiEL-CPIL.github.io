import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s safety margin for 60s limit
    
    # We want to maximize size. Try high sizes first.
    # The seed is 3. Observed best 7. Let's try 12 down to 3.
    
    best_result = [31, 32, 34] # Fallback
    
    for target_size in range(12, 2, -1):
        if time.time() > deadline:
            break
        result = _search(target_size, deadline)
        if result is not None:
            best_result = result
            # If we found a valid set, we can try to go bigger? 
            # The loop goes down, so we found the max so far. 
            # We can break if we found a large one, but let's try to find even bigger in next iter? 
            # No, loop is descending. So first success is the max.
            # But maybe a slightly smaller target is found faster? 
            # Usually, larger sets are harder. If 12 fails fast, 11 is tried.
            # Once we find a valid set for target_size, we return it? 
            # The task says "Maximize set size". So we should return the largest found.
            # Since we iterate from high to low, the first valid one is the largest.
            return result
            
    return best_result

def _search(target_size, deadline):
    # Randomized greedy with backtracking
    # To avoid getting stuck in bad local minima, we can randomize candidate order
    
    # Initial candidates: 1..256
    # Heuristic: Start with numbers that are "far apart" to avoid small sum collisions/divisibility?
    # Or start with small numbers to allow more numbers?
    # Let's try a mix.
    
    def dfs(current_set, sums, depth):
        if time.time() > deadline:
            return None
        
        if depth == target_size:
            return current_set.copy()
        
        # Generate candidates
        # We can filter candidates that are already in current_set
        # To improve success rate, we might want to try candidates that don't create immediate divisibility with existing sums.
        
        # Simple heuristic: Try candidates in random order? Or sorted?
        # Sorted ascending might be bad because small numbers create dense sums.
        # Sorted descending might be better for divisibility (sparse sums).
        # Let's try descending order for candidates.
        
        candidates = [i for i in range(256, 0, -1) if i not in current_set]
        
        # Optional: Shuffle to add randomness if we want to explore different branches
        # But deterministic descending is a good start.
        
        for c in candidates:
            # Check if adding c is valid
            new_sums = set(sums)
            new_sums.add(c)
            valid = True
            
            for s in sums:
                ns = s + c
                if ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if not valid:
                continue
            
            # Check divisibility in new_sums
            # For each v in new_sums, check if any u in new_sums (u < v) divides v
            # Optimization: Iterate u, check multiples
            for u in new_sums:
                if u == 1:
                    valid = False
                    break
                multiple = u * 2
                while multiple <= max(new_sums):
                    if multiple in new_sums:
                        valid = False
                        break
                    multiple += u
                if not valid:
                    break
            
            if valid:
                # Recurse
                current_set.append(c)
                result = dfs(current_set, new_sums, depth + 1)
                if result is not None:
                    return result
                current_set.pop()
                
        return None

    # Start with empty set
    return dfs([], {0}, 0) # Wait, sums should start empty? 
    # Non-empty subset sums. 
    # If current_set is empty, sums is empty. 
    # Adding c: sums becomes {c}. 
    # My logic: 
    # dfs([], set(), 0)
    # c=256. new_sums = {256}. Valid? 256 doesn't divide anything else. 
    # Next c=255. new_sums = {256, 255, 511}. 
    # Check 255 | 511? No. 256 | 511? No. 
    # Check 255 | 256? No. 
    # So {256, 255} is valid.
    
    # Note: The sum 0 is not a non-empty subset sum. So we don't include 0 in the divisibility check.
    # My `sums` variable holds the non-empty subset sums of `current_set`.
    
    # Re-evaluating `_search` initialization
    # If I pass `{0}`, then `s+c` includes `0+c = c`.
    # And `new_sums` will contain `c` from `s=0` and `c` from `add(c)`. Duplicate!
    # So `sums` should be the set of non-empty subset sums.
    # Initially empty set.
    
    # Corrected DFS:
    def dfs(current_set, sums, depth):
        if time.time() > deadline:
            return None
        
        if depth == target_size:
            return current_set.copy()
        
        candidates = [i for i in range(256, 0, -1) if i not in current_set]
        
        for c in candidates:
            new_sums = set(sums)
            new_sums.add(c)
            valid = True
            
            for s in sums:
                ns = s + c
                if ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if not valid:
                continue
            
            # Divisibility check
            # We only care about pairs of non-empty sums.
            for u in new_sums:
                if u == 1:
                    valid = False
                    break
                multiple = u * 2
                # Max possible sum is sum(current_set) + c
                # But checking up to max(new_sums) is fine
                max_sum = max(new_sums)
                while multiple <= max_sum:
                    if multiple in new_sums:
                        valid = False
                        break
                    multiple += u
                if not valid:
                    break
            
            if valid:
                current_set.append(c)
                result = dfs(current_set, new_sums, depth + 1)
                if result is not None:
                    return result
                current_set.pop()
                
        return None

    return dfs([], set(), 0)
