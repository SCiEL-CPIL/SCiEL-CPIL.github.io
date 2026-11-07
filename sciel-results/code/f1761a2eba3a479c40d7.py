"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import sys
import time
from itertools import combinations

P = {'n': 256, 'maximum_size': 30}

def construct():
    """
    Find a large set of distinct integers in 1..256 such that:
    1. All nonempty subset sums are distinct.
    2. No two distinct subset sums divide one another.
    
    Strategy: Branch and Bound / Backtracking search.
    - Candidates are integers 1..256.
    - We try to build a set S.
    - State: current elements, current subset sums.
    - Pruning: Check divisibility constraints on the new set of sums.
    - Heuristic: Try larger numbers first to keep sums large and reduce divisibility conflicts.
    """
    
    # Global timeout
    START_TIME = time.time()
    TIMEOUT = 55  # Leave 5s margin
    
    # Global best
    best_set = []
    
    # Candidates: 256 down to 1
    candidates = list(range(256, 0, -1))
    
    def check_divisibility(sums):
        """
        Check if any two distinct sums in 'sums' divide each other.
        Returns True if valid (no division), False otherwise.
        """
        # Convert to sorted list for efficient checking
        sorted_sums = sorted(sums)
        n = len(sorted_sums)
        
        for i in range(n):
            u = sorted_sums[i]
            # If u is large, its multiples might exceed the max sum
            # Max sum is sorted_sums[-1]
            if u * 2 > sorted_sums[-1]:
                break
            
            # Check if any v in sorted_sums[i+1:] is a multiple of u
            # We can iterate
            for j in range(i + 1, n):
                v = sorted_sums[j]
                if v % u == 0:
                    return False
                # Optional optimization: if v > u * (max_multiple_possible), break?
                # Not straightforward without knowing max multiple.
                
        return True

    def dfs(current_elements, current_sums, start_idx):
        nonlocal best_set, START_TIME, TIMEOUT
        
        if time.time() - START_TIME > TIMEOUT:
            return
        
        # Update best
        if len(current_elements) > len(best_set):
            best_set = list(current_elements)
            
        # If we reached max size or no more candidates
        if len(current_elements) >= 30:
            return
            
        # Try to add more elements
        # We iterate from start_idx to end
        # To improve search, we might want to try all candidates, but 
        # using start_idx ensures we don't pick smaller indices again (combinations)
        
        for i in range(start_idx, len(candidates)):
            cand = candidates[i]
            
            # Generate new sums
            # new_sums = current_sums U {x} U {s+x for s in current_sums}
            
            # Check distinctness first (cheap)
            # x must not be in current_sums
            if cand in current_sums:
                continue
                
            new_sums = set(current_sums)
            new_sums.add(cand)
            
            # Add shifted sums
            valid_distinct = True
            for s in current_sums:
                ns = s + cand
                if ns in new_sums:
                    valid_distinct = False
                    break
                new_sums.add(ns)
                
            if not valid_distinct:
                continue
                
            # Check divisibility (expensive)
            if check_divisibility(new_sums):
                dfs(current_elements + [cand], new_sums, i + 1)
                
                # Pruning: If we found a set of size 30, we can stop? 
                # Or if we are close to timeout.
                if time.time() - START_TIME > TIMEOUT:
                    return

    # Initialize
    dfs([], {0}, 0) # Note: subset sums usually include 0 for empty set. 
    # But task says "NONEMPTY subset sum". 
    # However, for distinctness and divisibility logic:
    # If we include 0, 0 divides nothing? 0 is not a divisor. 
    # But 0 is not a non-empty sum. 
    # Let's stick to non-empty sums in the set passed to check_divisibility?
    # If I pass {0}, and add x, sums become {0, x, x}. 
    # 0 is not a non-empty sum. 
    # So I should NOT include 0 in the sums set used for checking.
    
    # Correction:
    # Start with empty set of non-empty sums.
    # When adding x, new sums are {x} U {s+x | s in current_sums}.
    
    # Let's restart the logic with this correction.
    
    def check_divisibility_nonempty(sums):
        if not sums:
            return True
        sorted_sums = sorted(sums)
        n = len(sorted_sums)
        for i in range(n):
            u = sorted_sums[i]
            if u * 2 > sorted_sums[-1]:
                break
            for j in range(i + 1, n):
                v = sorted_sums[j]
                if v % u == 0:
                    return False
        return True

    def dfs_correct(current_elements, current_sums, start_idx):
        nonlocal best_set, START_TIME, TIMEOUT
        
        if time.time() - START_TIME > TIMEOUT:
            return
        
        if len(current_elements) > len(best_set):
            best_set = list(current_elements)
            
        if len(current_elements) >= 30:
            return
            
        for i in range(start_idx, len(candidates)):
            cand = candidates[i]
            
            # Check if cand is already a sum (distinctness)
            if cand in current_sums:
                continue
                
            new_sums = set(current_sums)
            new_sums.add(cand)
            
            valid_distinct = True
            for s in current_sums:
                ns = s + cand
                if ns in new_sums:
                    valid_distinct = False
                    break
                new_sums.add(ns)
                
            if not valid_distinct:
                continue
                
            if check_divisibility_nonempty(new_sums):
                dfs_correct(current_elements + [cand], new_sums, i + 1)
                
                if time.time() - START_TIME > TIMEOUT:
                    return

    # Start search
    # Initial sums is empty set
    dfs_correct([], set(), 0)
    
    # Ensure we return a valid list
    if not best_set:
        # Fallback to a known small valid set if search fails completely
        # {3, 5} is valid? Sums: 3, 5, 8. 3|? No. 5|? No. 8|? No.
        best_set = [3, 5]
        
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
