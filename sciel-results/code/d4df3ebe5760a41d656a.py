import time
import numpy as np

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # We want to select up to 48 distinct positive integers from [1, 512]
    # such that no element is the mean of any subset of >= 2 other elements.
    
    # Strategy: Greedy with smart ordering and efficient subset-sum DP.
    # Order candidates to maximize density.
    
    # Key insight: The constraint 2a = b + c (for pairs) is the most restrictive.
    # If we avoid 3-term arithmetic progressions, we handle the k=2 case.
    # For larger k, the constraint is weaker (harder to satisfy the mean condition).
    
    # Let's use a greedy approach: try to add elements in an order that
    # minimizes conflicts.
    
    # Efficient check: Given a set S and a candidate a, check:
    # 1. For a: is there a subset of S with size >= 2 and sum = k*a for some k?
    # 2. For each x in S: is there a subset of (S \ {x}) ∪ {a} with size >= 2 
    #    and sum = k*x for some k?
    
    # For efficiency, we precompute subset-sum info incrementally.
    
    # Let's try a different approach: start with a known good structure.
    # Powers of 2 are "mean-free" in a strong sense, but we only get ~9 in [1,512].
    # 
    # Better: use a greedy that processes candidates and uses bitset DP.
    
    # Precompute: for a set of numbers, maintain a 2D bitset:
    # dp[sum] = bitmask of achievable subset sizes
    # But sum can be up to 48*512 = 24576, and sizes up to 48.
    # That's a 24576 x 48 bit matrix = manageable with numpy.
    
    # Actually, let's think differently. The maximum sum of any subset is at most
    # 48 * 512 = 24576. For each sum, we track which subset sizes are achievable.
    
    # dp[s] is an integer where bit k is set if sum s is achievable with exactly k elements.
    
    def check_valid(selected, candidate):
        """Check if adding candidate to selected maintains the constraint."""
        # For the new element 'candidate': check subsets of selected
        # For each existing element x: check subsets of (selected \ {x}) ∪ {candidate}
        
        all_elements = selected + [candidate]
        
        # For each element a in all_elements, check if any subset of OTHER elements
        # has sum = k * a for some k >= 2
        
        for a in all_elements:
            others = [x for x in all_elements if x != a]
            n_others = len(others)
            if n_others < 2:
                continue
            
            # Compute subset-sum DP for 'others'
            # dp[s] = bitmask of subset sizes achieving sum s
            max_sum = sum(others)
            # We only care about sums that are multiples of 'a' with k >= 2
            # i.e., sums = 2a, 3a, ..., n_others*a
            # And we need to check if those sums are achievable with the corresponding k
            
            # Use bitset DP
            # dp is a list of integers, dp[s] has bit k set if sum s achievable with k elements
            dp = [0] * (max_sum + 1)
            dp[0] = 1  # empty subset, size 0
            
            for x in others:
                # Process in reverse to avoid using x multiple times
                for s in range(max_sum, x - 1, -1):
                    if dp[s - x] != 0:
                        # Shift the size bitmask by 1
                        dp[s] |= (dp[s - x] << 1)
            
            # Now check: for k = 2 to n_others, is bit k set in dp[k*a]?
            for k in range(2, n_others + 1):
                target_sum = k * a
                if target_sum <= max_sum:
                    if (dp[target_sum] >> k) & 1:
                        return False
                else:
                    break  # larger k will only have larger target_sum
            
        return True
    
    # Greedy: try candidates in different orders
    # Start with a heuristic: maybe process in a way that spreads out values
    
    # Let's try multiple strategies and pick the best
    best_result = []
    
    # Strategy 1: ascending order
    # Strategy 2: descending order  
    # Strategy 3: interleaved (evens then odds, or some pattern)
    
    strategies = [
        list(range(1, 513)),           # ascending
        list(range(512, 0, -1)),      # descending
        list(range(1, 513, 2)) + list(range(2, 513, 2)),  # odds then evens
        list(range(2, 513, 2)) + list(range(1, 513, 2)),  # evens then odds
    ]
    
    for strategy in strategies:
        if time.time() > deadline:
            break
        
        selected = []
        for x in strategy:
            if len(selected) >= 48:
                break
            if time.time() > deadline:
                break
            if check_valid(selected, x):
                selected.append(x)
        
        if len(selected) > len(best_result):
            best_result = selected[:]
        
        if len(best_result) >= 48:
            break
    
    # If we didn't reach 48, try to improve with local search
    if len(best_result) < 48:
        # Try to add more elements
        # Or try a different initial set
        
        # Alternative: start with a sparse set and build up
        # Use a more systematic approach
        pass
    
    return best_result
