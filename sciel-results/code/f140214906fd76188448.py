# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit
    
    candidates = list(range(1, 23))
    
    # First, run the simple greedy to get a baseline (current best 16)
    current_set = []
    current_sums = {Fraction(0, 1)}
    for c in candidates:
        if time.time() > deadline:
            break
        added = Fraction(1, c)
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + added)
        if len(new_sums) == len(current_sums) * 2:
            current_set.append(c)
            current_sums = new_sums
            
    best_set = current_set[:]
    best_size = len(best_set)
    
    # Try to find a larger set using DFS
    # We try to find a set of size K, starting from 17 downwards
    # If we find one, we return it immediately as it's better than 16
    
    for target_size in range(17, best_size + 1, -1):
        if time.time() > deadline - 5: # Leave 5s for cleanup
            break
            
        found_set = None
        # Use DFS to find a valid set of size target_size
        # We can use a recursive function
        
        def dfs(idx, cur_set, cur_sums):
            nonlocal found_set
            if found_set is not None:
                return True
            if time.time() > deadline - 5:
                return False
            if len(cur_set) == target_size:
                found_set = cur_set[:]
                return True
            if idx >= len(candidates):
                return False
            # Pruning: if not enough candidates left to reach target_size
            if len(cur_set) + (len(candidates) - idx) < target_size:
                return False
            
            c = candidates[idx]
            
            # Option 1: Include c
            added = Fraction(1, c)
            new_sums = set(cur_sums)
            for s in cur_sums:
                new_sums.add(s + added)
            if len(new_sums) == len(cur_sums) * 2:
                if dfs(idx + 1, cur_set + [c], new_sums):
                    return True
            
            # Option 2: Exclude c
            if dfs(idx + 1, cur_set, cur_sums):
                return True
                
            return False
        
        # Initialize DFS
        # Heuristic: Try including small numbers first?
        # The candidates are already sorted 1..22.
        # DFS will try to include 1, then 2, etc.
        # This might be slow if 1, 2, 3... don't lead to a solution of size 17.
        # But it's a good start.
        
        dfs(0, [], {Fraction(0, 1)})
        
        if found_set:
            best_set = found_set
            best_size = target_size
            break # Found max possible for this target, can stop or try higher? 
                   # We are iterating down, so we found the largest K >= best_size.
                   # Actually, we iterate 17, 16... 
                   # If we find 17, we return 17.
                   # If we don't find 17, we try 16 (which we already have).
                   # So we can break if found_size == 17.
                   # Or just return best_set at the end.
                   
    return best_set

# EVOLVE-BLOCK-END
