"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    
    Uses a greedy approach with bitset DP to try to find a set of size 32.
    Falls back to the arithmetic progression {226..256} of size 31 if 32 is not found.
    """
    import random
    
    def is_valid(elements):
        """
        Check if the set of elements satisfies the cardinality-separated subset sum condition.
        Uses bitset DP where dp[k] is an integer with bit s set if sum s is achievable with k elements.
        """
        n = len(elements)
        max_sum = sum(elements)
        # dp[k] is a bitset of achievable sums with k elements
        # Use list of integers as bitsets
        dp = [0] * (n + 1)
        dp[0] = 1  # sum 0 is achievable with 0 elements
        
        for val in elements:
            # Update dp in reverse to avoid using the same element twice
            for k in range(n, 0, -1):
                # Shift dp[k-1] by val and OR into dp[k]
                dp[k] |= (dp[k-1] << val)
        
        # Check for collisions between different cardinalities
        for k in range(1, n + 1):
            for m in range(k + 1, n + 1):
                if dp[k] & dp[m]:
                    return False
        return True
    
    def greedy_search(target_size, start_range, attempts=100):
        """
        Try to find a valid set of target_size using greedy with randomization.
        """
        for _ in range(attempts):
            current = []
            # Shuffle the candidate range
            candidates = list(start_range)
            random.shuffle(candidates)
            
            for val in candidates:
                if len(current) >= target_size:
                    break
                # Try adding val
                test_set = current + [val]
                if is_valid(test_set):
                    current.append(val)
            
            if len(current) >= target_size:
                return current
        
        return None
    
    # Try to find a set of size 32
    # Strategy: try greedy with different orderings
    # First, try the AP set as a baseline
    ap_set = list(range(226, 257))  # size 31
    
    # Try to extend or find a better set
    # Use greedy from high to low
    found = greedy_search(32, range(256, 1, -1), attempts=50)
    if found:
        return found
    
    # Try from low to high
    found = greedy_search(32, range(1, 257), attempts=50)
    if found:
        return found
    
    # Try mixed approach: start with high numbers
    found = greedy_search(32, range(200, 257), attempts=100)
    if found:
        return found
    
    # Fall back to the AP set of size 31
    return ap_set

# EVOLVE-BLOCK-END
