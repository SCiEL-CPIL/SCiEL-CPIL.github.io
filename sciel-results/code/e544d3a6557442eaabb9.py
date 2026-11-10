"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 256, 'maximum_size': 60}

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ. Repeated sums within the
    SAME cardinality are allowed. Maximize set size.
    
    Uses global conflict-driven removal with exact cardinality-indexed subset-sum DP.
    """
    n_max = 256
    target_max_size = 60
    timeout_seconds = 55  # Leave 5s margin for the 60s limit
    
    start_time = time.time()
    
    def is_valid_fast(S):
        """Check if set S satisfies the cross-cardinality sum disjointness property."""
        m = len(S)
        if m == 0:
            return True
        
        dp = [0] * (m + 1)
        dp[0] = 1  # bit 0 is set for cardinality 0
        
        for x in S:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
        
        # Check for collisions between different cardinalities
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                if dp[i] & dp[j]:
                    return False
        return True
    
    def count_collisions(S):
        """Count total number of cross-cardinality sum collisions."""
        m = len(S)
        if m == 0:
            return 0
        
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in S:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
        
        total_collisions = 0
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                total_collisions += bin(dp[i] & dp[j]).count('1')
        return total_collisions
    
    def find_best_removal(S):
        """
        Find the element whose removal reduces collisions the most.
        Returns (element, new_collision_count) or (None, None) if S is empty.
        """
        m = len(S)
        if m == 0:
            return None, None
        
        current_collisions = count_collisions(S)
        if current_collisions == 0:
            return None, 0  # Already valid
        
        best_element = None
        min_collisions_after = current_collisions + 1  # Start with worst possible
        
        # Try removing each element
        for idx, x in enumerate(S):
            new_S = S[:idx] + S[idx+1:]
            c = count_collisions(new_S)
            if c < min_collisions_after:
                min_collisions_after = c
                best_element = x
                if c == 0:
                    break  # Can't do better than 0
        
        return best_element, min_collisions_after
    
    def construct_with_removal(initial_size, time_limit):
        """
        Start with a random set of given size, then iteratively remove elements
        involved in the most frequent collisions until valid or time runs out.
        """
        candidates = list(range(1, n_max + 1))
        random.shuffle(candidates)
        S = candidates[:initial_size]
        
        # If already valid, return
        if is_valid_fast(S):
            return S
        
        # Iterative removal
        max_iterations = initial_size + 5
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Check if currently valid
            if is_valid_fast(S):
                break
            
            # Find best element to remove
            best_elem, new_coll = find_best_removal(S)
            
            if best_elem is None:
                break
            
            # Remove the element
            S = [x for x in S if x != best_elem]
            
            # If we've removed too many, stop
            if len(S) < 2:
                break
        
        return S
    
    # Try to construct a valid set, starting with a large size and working down
    # We'll try several initial sizes and keep the best valid result
    
    best_result = []
    
    # First, try starting with size 60
    for initial_size in [60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 8]:
        if time.time() - start_time > time_limit - 5:
            break
        
        result = construct_with_removal(initial_size, time_limit - 5)
        
        if len(result) > len(best_result) and is_valid_fast(result):
            best_result = result
        
        # If we found a valid set of size 60, we're done
        if len(best_result) == 60:
            break
    
    # If we still don't have a good result, try a simpler greedy approach as fallback
    if len(best_result) < 10:
        S = []
        used = set()
        for x in range(1, n_max + 1):
            if time.time() - start_time > time_limit:
                break
            new_S = S + [x]
            if is_valid_fast(new_S):
                S = new_S
                used.add(x)
        
        if len(S) > len(best_result):
            best_result = S
    
    # Final safety check
    if not is_valid_fast(best_result):
        # If somehow invalid, return empty set (valid by definition)
        best_result = []
    
    return list(best_result)

# EVOLVE-BLOCK-END

