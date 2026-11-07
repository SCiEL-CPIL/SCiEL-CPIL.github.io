import time
import random

def _can_add(dp, union_dp, x, n):
    """
    Check if x can be added to the current set represented by dp.
    dp[k] is a bitset of sums achievable with exactly k elements.
    union_dp is the OR of all dp[k].
    n is the current number of elements.
    """
    # We need to check if any new sum (from adding x to a subset of size k-1)
    # collides with an existing sum from a subset of size j != k.
    # New sums for cardinality k are (dp[k-1] << x).
    # Existing sums for cardinality j are dp[j].
    # Collision if (dp[k-1] << x) & dp[j] != 0 for any j != k.
    
    # Optimization: 
    # Let U = union_dp.
    # Forbidden bits for new sums of cardinality k are U & ~dp[k].
    # If (dp[k-1] << x) & (U & ~dp[k]) != 0, then collision.
    
    for k in range(1, n + 2):
        # dp has length n+1 (indices 0..n).
        # dp[k-1] is valid for k-1 <= n => k <= n+1.
        # dp[k] is valid for k <= n. If k == n+1, dp[k] is out of bounds, treat as 0.
        new_sums = dp[k - 1] << x
        if k < len(dp):
            existing_k = dp[k]
        else:
            existing_k = 0
        forbidden = union_dp & ~existing_k
        if new_sums & forbidden:
            return False
    return True

def _add_element(dp, union_dp, x, n):
    """
    Add x to the set, updating dp and union_dp.
    Returns new_dp, new_union_dp.
    """
    new_dp = [0] * (n + 2)
    new_dp[0] = 1 # Empty set sum 0
    
    # Update sums
    # new_dp[k] = old_dp[k] | (old_dp[k-1] << x)
    for k in range(1, n + 2):
        new_dp[k] = dp[k] | (dp[k - 1] << x)
        
    # Update union
    new_union = 0
    for k in range(n + 2):
        new_union |= new_dp[k]
        
    return new_dp, new_union

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin
    
    best_size = 0
    best_set = []
    
    # Strategy 1: Greedy from 1 to 256
    dp = [1]
    union_dp = 1
    n = 0
    current_set = []
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if _can_add(dp, union_dp, x, n):
            dp, union_dp = _add_element(dp, union_dp, x, n)
            current_set.append(x)
            n += 1
            
    if n > best_size:
        best_size = n
        best_set = current_set
        
    # Strategy 2: Random Restarts
    # We try to find a better ordering
    attempts = 0
    while time.time() < deadline - 5.0: # Leave 5s buffer
        attempts += 1
        # Generate a random permutation
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        dp = [1]
        union_dp = 1
        n = 0
        current_set = []
        
        for x in candidates:
            if time.time() > deadline:
                break
            if _can_add(dp, union_dp, x, n):
                dp, union_dp = _add_element(dp, union_dp, x, n)
                current_set.append(x)
                n += 1
                
        if n > best_size:
            best_size = n
            best_set = current_set
            
        # Early stop if we hit max possible? 
        # Max size is 60. 
        # Also, if we reach a size that seems very good, we can break.
        if best_size >= 40: # Arbitrary threshold, likely won't reach this high
             break

    # Ensure return type is list of ints and non-empty
    if not best_set:
        return [1]
    
    # Convert to Python int to satisfy type checks
    return [int(x) for x in best_set]
