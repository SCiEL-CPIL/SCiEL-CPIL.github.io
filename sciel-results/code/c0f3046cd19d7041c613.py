import time
from itertools import combinations
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    # LCM of 1 to 22
    L = reduce(lcm, range(1, 23))
    
    # Precompute values L // i
    vals = [L // i for i in range(1, 23)]
    
    best_subset = []
    best_size = 0
    
    # We try to find the largest k
    # We know 16 is possible. Let's start from 22 down to 1.
    # However, searching for k=22, 21, etc. might be quick to fail or succeed.
    # If we find a solution for k, we return it immediately?
    # The task says "Maximize cardinality". So we must find the max.
    # If we find a solution for k=22, that's the max.
    # If we find a solution for k=16, we must check if 17, 18... 22 are possible.
    # So we iterate k from 22 down to 1.
    
    for k in range(22, 0, -1):
        if time.time() > deadline:
            break
            
        # If we already have a solution of size >= k, we can stop?
        # No, we are iterating down. If we are at k=16 and have best_size=16, 
        # we are checking if 16 is possible. If we find one, we update best.
        # But we want the MAX. So if we find a solution for k=17, we return it.
        # If we find a solution for k=16, we continue to check if 17 is possible?
        # Wait, we iterate k from 22 down to 1.
        # So we check 22 first. If found, return.
        # Then 21. If found, return.
        # ...
        # Then 16. If found, we don't return immediately because we haven't checked 17..22?
        # No, we already checked 22..17. If they failed, then 16 is the max.
        # So if we find a solution for k, it is the maximum possible cardinality 
        # because we checked all larger k's and found no solution.
        
        # Search for a dissociated subset of size k
        found = _search_max_k(k, vals, deadline)
        if found:
            # Convert indices back to values
            subset = [i + 1 for i in found]
            return subset
            
    # If loop finishes without finding a specific k (due to time or no solution),
    # return the best found so far? 
    # In the logic above, if _search_max_k returns a solution, we return it.
    # If it returns None, we continue to k-1.
    # If we run out of time, we break.
    # We should keep track of the best subset found in the process?
    # Actually, if we are checking k=16 and it times out, we don't have a solution for 16.
    # But we might have found one in a previous iteration? No, previous iterations were larger k.
    # If larger k failed, we don't have a solution for them.
    # So if we time out, we might have no solution?
    # But we know 16 is possible. We should ensure we find it.
    # To be safe, we can hardcode a known solution if we fail?
    # Or, we can just return the seed solution if we fail?
    # The seed solution is size 9.
    
    # Let's refine the search.
    # If we don't find a solution for k=22 down to k+1, and we are checking k,
    # and we time out, we don't know if k is possible.
    # However, if we find a solution for k, it is the max.
    
    # Fallback: return a known valid subset
    # The seed returns [1, 2, 3, 5, 7, 11, 13, 17, 19]
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]

def _search_max_k(k, vals, deadline):
    """
    Search for a dissociated subset of size k from indices 0..21 (values 1..22).
    Returns the list of indices if found, else None.
    """
    # Backtracking
    # We can use a recursive function or iterative.
    # Given k can be up to 22, recursion depth is fine.
    
    # Optimization: Order of selection.
    # Does it matter? Not really for correctness, but for speed.
    # We can try to pick elements that are "large" to avoid collisions?
    # Or just iterate combinations.
    
    # For small k (like 22, 21), combinations are few.
    # For k=16, combinations are ~75k.
    # We can just iterate combinations and check.
    
    # Check function
    def is_dissociated(indices):
        sums = {0}
        for idx in indices:
            v = vals[idx]
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    # Iterate combinations
    # To speed up, we can use pruning?
    # For now, simple iteration.
    
    # If k is large, the number of combinations is small.
    # If k is small, the number is large, but we only reach small k if larger ones fail.
    
    for comb in combinations(range(22), k):
        if time.time() > deadline:
            return None
        if is_dissociated(comb):
            return list(comb)
            
    return None
