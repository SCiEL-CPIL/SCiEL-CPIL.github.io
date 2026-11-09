import time
from math import lcm
from functools import reduce
from itertools import combinations

# Universe is [1, 24]
UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)

# Precompute weights
WEIGHTS = {x: L // x for x in UNIVERSE}

def get_subset_sums(S):
    """
    Return a set of all possible sums of subsets of S.
    S is a list of integers (weights).
    """
    sums = {0}
    for w in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        sums.update(new_sums)
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # We want to maximize the size.
    # Let's try to find a set of size k.
    # We can iterate k from 24 down to 1 and check if a valid set of size k exists.
    # Or we can use DFS to find the largest valid set directly.
    
    # DFS approach with branch and bound.
    # Order: Try adding elements in descending order of value (ascending order of weight).
    # This heuristic works well because large values (small weights) are "independent" 
    # in the sense that their reciprocals are small and hard to form from other distinct reciprocals 
    # unless the set is very large and dense.
    
    # Let's sort candidates by value descending
    candidates = sorted(UNIVERSE, reverse=True)
    
    best_size = 0
    best_set = []
    
    # To speed up, we can precompute subset sums for prefixes? 
    # No, the subset depends on the path.
    
    # We will use a recursive backtracking search.
    # State: current_index, current_set (list of values), current_weights (list of L/x)
    
    # Optimization: Memoization is hard because the state is the set itself.
    # However, the branching factor is small if we prune well.
    
    # Let's add a time check helper
    def time_exceeded():
        return time.time() > deadline

    def dfs(idx, current_values, current_weights, current_sums):
        nonlocal best_size, best_set
        
        if time_exceeded():
            return
        
        # Update best
        if len(current_values) > best_size:
            best_size = len(current_values)
            best_set = current_values[:]
            
        # Pruning: Upper bound on size
        # We can add at most (len(candidates) - idx) more elements.
        if len(current_values) + (len(candidates) - idx) <= best_size:
            return
            
        # If we've considered all candidates
        if idx == len(candidates):
            return
        
        # Option 1: Skip candidates[idx]
        dfs(idx + 1, current_values, current_weights, current_sums)
        
        if time_exceeded():
            return
            
        # Option 2: Include candidates[idx]
        x = candidates[idx]
        w = WEIGHTS[x]
        
        # Check if w is in current_sums (formed by other selected elements)
        # Note: current_sums contains sums of subsets of current_weights.
        # We need to check if w can be formed by a subset of current_weights.
        # Since all weights are positive, if w is in current_sums, it's a conflict.
        # Note: The sum must be of a subset of OTHER elements. 
        # current_sums includes 0 and sums of subsets of current_weights.
        # If w is in current_sums, it means there exists a subset T of current_values such that sum(1/y for y in T) = 1/x.
        # This is exactly the forbidden condition.
        
        if w in current_sums:
            # Conflict, cannot include x
            return
        
        # Include x
        new_values = current_values + [x]
        new_weights = current_weights + [w]
        
        # Update subset sums
        # new_sums = current_sums U {s + w for s in current_sums}
        new_sums = set(current_sums)
        for s in current_sums:
            new_sums.add(s + w)
            
        dfs(idx + 1, new_values, new_weights, new_sums)

    # Initialize
    dfs(0, [], [], {0})
    
    # If DFS didn't find anything (shouldn't happen, single element is always valid)
    if not best_set:
        best_set = [1]
        
    return best_set

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(f"Best set size: {len(s)}")
    print(f"Set: {s}")
    
    # Verify
    L_check = reduce(lcm, range(1, 25))
    valid = True
    for x in s:
        target = L_check // x
        others = [L_check // y for y in s if y != x]
        # Check if target is subset sum of others
        sums = {0}
        for w in others:
            new_sums = set()
            for u in sums:
                new_sums.add(u + w)
            sums.update(new_sums)
        if target in sums:
            valid = False
            print(f"Invalid: {x} is sum of others")
    print(f"Valid: {valid}")
