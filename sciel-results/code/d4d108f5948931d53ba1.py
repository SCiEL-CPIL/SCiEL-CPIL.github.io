import time
from fractions import Fraction
import random

def check_conflict(S, x):
    """
    Check if 1/x is a subset sum of {1/y for y in S if y > x}.
    Returns the subset Y if found, else None.
    """
    # Candidates must be > x
    candidates = [y for y in S if y > x]
    if not candidates:
        return None
    
    target = Fraction(1, x)
    
    # Pruning: max possible sum
    max_sum = sum(Fraction(1, y) for y in candidates)
    if max_sum < target:
        return None
    
    # Sort candidates by value descending (1/y descending -> y ascending)
    candidates.sort()
    
    def dfs(idx, current_sum, current_subset):
        if current_sum == target:
            return current_subset
        if idx == len(candidates):
            return None
        if current_sum > target:
            return None
        
        # Option 1: Skip
        res = dfs(idx + 1, current_sum, current_subset)
        if res:
            return res
            
        # Option 2: Take
        y = candidates[idx]
        new_sum = current_sum + Fraction(1, y)
        if new_sum <= target:
             res = dfs(idx + 1, new_sum, current_subset | {y})
             if res:
                 return res
                 
        return None

    return dfs(0, Fraction(0), set())

def is_valid(S):
    for x in S:
        if check_conflict(S, x) is not None:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    # Initial set
    S = set(range(1, 25))
    
    # Heuristics for removal
    heuristics = [
        lambda cands: max(cands), # Remove largest index (smallest fraction)
        lambda cands: min(cands), # Remove smallest index (largest fraction)
        lambda cands: random.choice(list(cands))
    ]
    
    best_S = set()
    
    for h_idx, heuristic in enumerate(heuristics):
        if time.time() > deadline:
            break
            
        S = set(range(1, 25))
        random.seed(h_idx)
        
        while time.time() < deadline:
            # Find a conflict
            conflict_found = False
            for x in sorted(S):
                Y = check_conflict(S, x)
                if Y is not None:
                    # Conflict: x and Y
                    candidates_to_remove = {x} | Y
                    z_to_remove = heuristic(candidates_to_remove)
                    S.remove(z_to_remove)
                    conflict_found = True
                    break # Restart the loop with the new S
            
            if not conflict_found:
                break
        
        if len(S) > len(best_S):
            best_S = set(S)
            
    # Final check
    if not is_valid(best_S):
        best_S = {24}
        
    return list(best_S)
