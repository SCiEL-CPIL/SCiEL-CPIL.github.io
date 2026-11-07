import time

# EVOLVE-BLOCK-START

def _subset_sums_by_cardinality(elements):
    """
    Compute the set of achievable sums for each cardinality using DP.
    elements: list of distinct positive integers.
    Returns a list of sets, where index i is the set of sums for subsets of size i.
    """
    n = len(elements)
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    for x in elements:
        for i in range(n - 1, -1, -1):
            if dp[i]:
                new_sums = {s + x for s in dp[i]}
                dp[i + 1] |= new_sums
    return dp

def _is_valid(elements):
    """
    Check if the set of elements satisfies the cardinality-separated subset sum property.
    """
    if not elements:
        return True
    if len(set(elements)) != len(elements):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    
    dp = _subset_sums_by_cardinality(elements)
    n = len(elements)
    
    # Check disjointness of sum sets across different cardinalities
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if dp[i] and dp[j]:
                if dp[i] & dp[j]:
                    return False
    return True

def _greedy_construct(deadline):
    """
    Greedy algorithm to construct a valid set, trying to maximize size.
    Try adding elements in various orders, checking validity after each addition.
    """
    best = list(range(226, 257))  # Start with known valid set of size 31
    best_size = len(best)
    
    # Try different strategies for building larger sets
    strategies = [
        # Strategy 1: Start from known good set, try to add elements
        lambda: list(range(226, 257)),
        # Strategy 2: Start empty, add smallest valid elements first
        lambda: [],
        # Strategy 3: Start empty, add largest valid elements first  
        lambda: [],
    ]
    
    for start_idx, start_func in enumerate(strategies):
        if time.time() > deadline - 5:
            break
        
        current = start_func()
        current_set = set(current)
        
        # Determine candidate order
        if start_idx == 0:
            # Try adding elements not in current set
            candidates = [x for x in range(1, 257) if x not in current_set]
        elif start_idx == 1:
            candidates = list(range(1, 257))
        else:
            candidates = list(range(256, 0, -1))
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if x in current_set:
                continue
            
            test_list = current + [x]
            if _is_valid(test_list):
                current = test_list
                current_set.add(x)
        
        if len(current) > best_size:
            best = current
            best_size = len(current)
    
    return best

def construct():
    deadline = time.time() + 85  # Leave 5s safety margin
    result = _greedy_construct(deadline)
    
    # Ensure we return a valid result
    if not _is_valid(result):
        # Fallback to known valid set
        result = list(range(226, 257))
    
    return result

# EVOLVE-BLOCK-END
