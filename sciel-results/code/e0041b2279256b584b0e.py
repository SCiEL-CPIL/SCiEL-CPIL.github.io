"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

P = {'n': 256, 'maximum_size': 60}

def compute_dp(S):
    """
    Compute the set of achievable sums for each cardinality using bitsets.
    Returns a list of integers where the i-th integer is a bitset of sums achievable by subsets of size i.
    """
    m = len(S)
    if m == 0:
        return [1]  # dp[0] = bit 0 set
    
    dp = [0] * (m + 1)
    dp[0] = 1  # bit 0 is set for cardinality 0
    
    for x in S:
        # Iterate backwards to ensure each element is used at most once
        for k in range(m, 0, -1):
            # Shift the bitset for cardinality k-1 by x and OR into cardinality k
            dp[k] |= (dp[k-1] << x)
            
    return dp

def count_collisions(dp):
    """
    Count the total number of sum collisions across different cardinalities.
    A collision is a sum value that is achievable by subsets of two different cardinalities.
    """
    m = len(dp) - 1
    collisions = 0
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            common = dp[i] & dp[j]
            if common:
                # Use bit_count for efficiency (Python 3.8+)
                collisions += common.bit_count()
    return collisions

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ.
    
    Strategy:
    1. Start with a random set of size N.
    2. Compute global collision count.
    3. If collisions > 0, find the element whose removal minimizes the collision count.
    4. Remove that element.
    5. Repeat until valid or size is small.
    6. Use random restarts to escape local minima.
    """
    max_size = 60
    max_val = 256
    best_set = []
    best_score = -1
    
    # Try a few random restarts
    for restart in range(10):
        # Start with a random set of size max_size
        # We pick distinct integers from 1..256
        S = random.sample(range(1, max_val + 1), max_size)
        
        current_size = len(S)
        
        # Local search loop
        while current_size > 0:
            dp = compute_dp(S)
            colls = count_collisions(dp)
            
            if colls == 0:
                # Found a valid set
                if len(S) > len(best_set):
                    best_set = S.copy()
                    best_score = len(S)
                break
            
            # Find the best element to remove
            # We evaluate removing each element x in S
            min_colls = colls + 1
            best_x_to_remove = None
            
            for idx, x in enumerate(S):
                # Create a new set without x
                # To avoid creating a new list every time, we can just pass a slice or create a copy
                # Since size is small, copying is fine
                S_temp = S[:idx] + S[idx+1:]
                
                dp_temp = compute_dp(S_temp)
                colls_temp = count_collisions(dp_temp)
                
                if colls_temp < min_colls:
                    min_colls = colls_temp
                    best_x_to_remove = x
                    
                    # Early exit if we found a valid set by removing x
                    if colls_temp == 0:
                        break
            
            if best_x_to_remove is None:
                # Should not happen, but fallback
                break
                
            # Remove the best element
            S.remove(best_x_to_remove)
            current_size -= 1
            
            # If we removed an element, check if we found a valid set
            # The loop will check at the start of the next iteration
            # But if min_colls was 0, we can break early after removal?
            # Actually, the loop condition checks colls at the top.
            # If we removed x and it made it valid, the next iteration will see colls=0 and break.
            # To save time, we can check here.
            if min_colls == 0:
                 # We know S is now valid
                 if len(S) > len(best_set):
                     best_set = S.copy()
                     best_score = len(S)
                 break

    # Ensure we return a list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
