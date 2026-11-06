import time
import sys
import math

# Increase recursion limit just in case, though depth is ~20
sys.setrecursionlimit(10000)

def construct():
    """
    Find a maximum-size Sidon set (all pairwise sums a+b for a<=b distinct)
    containing 1,2,5,11 within [1,284].
    
    Uses Backtracking with Forward Checking and tighter pruning bounds.
    Includes a greedy initialization to find a good initial solution quickly.
    """
    prescribed = [1, 2, 5, 11]
    used_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            used_sums.add(x + y)
    
    current = list(prescribed)
    best_set = list(current)
    best_size = len(current)
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Helper to check if adding x is valid
    def is_valid(x, arr, used):
        # Check conflict with existing sums
        # new sums: x + y for y in arr, and 2x
        # Since x > max(arr), all x+y are distinct from each other and from 2x.
        # We just need to check against 'used'.
        for y in arr:
            if x + y in used:
                return False
        if 2 * x in used:
            return False
        return True

    def get_new_sums(x, arr):
        return {x + y for y in arr} | {2 * x}

    # Greedy initialization to find a decent initial solution
    def greedy_search():
        curr = list(prescribed)
        curr_used = set(used_sums)
        for x in range(12, 285):
            if is_valid(x, curr, curr_used):
                curr.append(x)
                curr_used.update(get_new_sums(x, curr))
        return curr

    # Run greedy to get a good starting point
    greedy_result = greedy_search()
    if len(greedy_result) > best_size:
        best_size = len(greedy_result)
        best_set = list(greedy_result)

    def backtrack(arr, used, start_val):
        nonlocal best_set, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return

        # Pruning with tighter bound:
        # The maximum number of elements we can add from [start_val, 284] 
        # is bounded by the max size of a Sidon set in an interval of length L = 284 - start_val + 1.
        # Max Sidon size in [1, L] is approx sqrt(2*L).
        # So max new elements k_new <= sqrt(2 * (284 - start_val + 1))
        remaining_range = 284 - start_val + 1
        if remaining_range > 0:
            max_possible_new = int(math.sqrt(2 * remaining_range)) + 1
        else:
            max_possible_new = 0
            
        if len(arr) + max_possible_new <= best_size:
            return

        if len(arr) > best_size:
            best_size = len(arr)
            best_set = list(arr)

        # Try adding each possible next element
        for x in range(start_val, 285):
            if is_valid(x, arr, used):
                new_used = used | get_new_sums(x, arr)
                backtrack(arr + [x], new_used, x + 1)

    # Start backtracking
    backtrack(current, used_sums, 12)
    
    return best_set
