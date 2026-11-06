# EVOLVE-BLOCK-START
import random
import time

def find_subset_sum_200(elements):
    """
    Finds a subset of 'elements' that sums to 200.
    Returns the list of elements in the subset, or None if no such subset exists.
    """
    target = 200
    dp = [False] * (target + 1)
    dp[0] = True
    parent = [None] * (target + 1)
    
    for x in elements:
        if x > target:
            continue
        for s in range(target, x - 1, -1):
            if dp[s - x] and not dp[s]:
                dp[s] = True
                parent[s] = (s - x, x)
                
    if not dp[target]:
        return None
        
    subset = []
    s = target
    while s > 0:
        prev_s, x = parent[s]
        subset.append(x)
        s = prev_s
    return subset

def construct():
    deadline = time.time() + 85
    best_set = []
    
    # Define heuristics for choosing which element to remove from a conflicting subset T
    # T is a list of elements summing to 200
    heuristics = [
        lambda T: max(T),          # Remove largest
        lambda T: min(T),          # Remove smallest
        lambda T: T[len(T)//2],    # Remove middle (after sorting? No, T is not sorted)
        lambda T: random.choice(T) # Remove random
    ]
    
    # We will run the local search with each heuristic and keep the best result
    for i, heuristic in enumerate(heuristics):
        if time.time() > deadline - 5:
            break
            
        current_S = set(range(1, 101))
        
        while time.time() < deadline - 5:
            # Convert to list for DP
            # Shuffling might help find different conflicts
            elements = list(current_S)
            random.shuffle(elements)
            
            T = find_subset_sum_200(elements)
            if T is None:
                break
                
            # Choose element to remove
            # For 'middle', we need to sort T first
            if i == 2:
                T_sorted = sorted(T)
                x = T_sorted[len(T_sorted)//2]
            else:
                x = heuristic(T)
                
            if x in current_S:
                current_S.remove(x)
                
        if len(current_S) > len(best_set):
            best_set = list(current_S)
            
    # If best_set is empty (should not happen as we start with 100 elements and remove one by one),
    # fallback to a valid set, e.g., {100, 99, ...} or just return [] if something went wrong
    # But the task requires a nonempty subset.
    if not best_set:
        # Fallback: A set with no subset summing to 200
        # e.g., {101, ..., 100} is empty. 
        # {100, 99, 98, ...} 
        # Let's just use the result of the first heuristic if it failed?
        # Actually, the loop above should always produce a valid set.
        # If it didn't run, maybe time was too short?
        # Let's do a quick greedy as fallback
        S = set(range(1, 101))
        while True:
            elements = list(S)
            T = find_subset_sum_200(elements)
            if T is None:
                break
            x = max(T)
            S.remove(x)
        best_set = list(S)
        
    return best_set
# EVOLVE-BLOCK-END
