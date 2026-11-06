import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    candidates = list(range(1, 23))
    L = reduce(lcm, candidates)
    
    # Precompute values
    vals = [L // c for c in candidates]
    
    def check_and_add(current_sums, val):
        """
        Returns new set of sums if adding val keeps them distinct, else None.
        current_sums: set of integers
        val: integer (L // i)
        """
        new_sums = set()
        for s in current_sums:
            ns = s + val
            if ns in current_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        return current_sums | new_sums

    # We search for the maximum K
    # K goes from 22 down to 1
    for K in range(22, 0, -1):
        if time.time() > deadline:
            break
            
        found_set = None
        
        def backtrack(idx, picked, current_set, current_sums):
            nonlocal found_set
            if time.time() > deadline:
                return False
                
            if picked == K:
                found_set = list(current_set)
                return True
                
            # Pruning: not enough candidates left to reach K
            if picked + (22 - idx) < K:
                return False
                
            # Option 1: Include candidates[idx]
            # Check if it fits
            val = vals[idx]
            new_sums = check_and_add(current_sums, val)
            if new_sums is not None:
                current_set.append(candidates[idx])
                if backtrack(idx + 1, picked + 1, current_set, new_sums):
                    return True
                current_set.pop()
                
            # Option 2: Exclude candidates[idx]
            if backtrack(idx + 1, picked, current_set, current_sums):
                return True
                
            return False

        if backtrack(0, 0, [], {0}):
            return found_set
            
    # Fallback (should not happen)
    return [1]
