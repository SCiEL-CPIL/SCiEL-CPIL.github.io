import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s
    
    # We'll use a greedy approach with smart ordering
    # Key insight: check violations efficiently using subset-sum properties
    
    # Precompute: for a set S, we need to know for each target t and each k,
    # whether there exists a subset of S of size k with sum = k*t
    # This is equivalent to: subset sum = k*t with exactly k elements
    
    # For efficiency, we'll maintain subset-sum information incrementally
    # But for 48 elements, we can do a full check each time with pruning
    
    # Strategy: process candidates in an order that tends to produce valid sets
    # Try multiple orderings and keep the best result
    
    def is_valid_set(S):
        """Check if set S has no element that is the mean of a subset of >=2 others."""
        if len(S) < 3:
            return True
        S_list = sorted(S)
        n = len(S_list)
        
        for i, a in enumerate(S_list):
            # Check if a is the mean of any subset of S_list excluding a, with size >= 2
            others = S_list[:i] + S_list[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            # For each k from 2 to m, check if there's a subset of others of size k with sum = k*a
            # Use DP: dp[k][s] = whether we can pick k elements from others summing to s
            # But s can be up to k*a which is large. Instead, use a different approach.
            
            # Alternative: for each k, we need subset of size k with sum exactly k*a
            # This is a "exact k-sum" problem. For small m (up to 47), we can use bitset DP.
            
            # dp[j] = bitset where bit s is set if we can pick j elements summing to s
            # But s ranges from 0 to k*a which could be large.
            
            # Better: for fixed k, check if there's a subset of size k with sum = k*a
            # Use meet-in-the-middle or DP with bounded sums
            
            # Since others elements are in [1,512] and k <= 47, max sum = 47*512 = 24064
            # We can use a DP table: dp[k][s] for k up to m and s up to max_sum
            # But that's O(m * max_sum) which is 47 * 24064 ≈ 1.1M, acceptable for 48 checks
            
            max_sum_needed = k_max = m
            # Actually let's do it per k
            for k in range(2, m + 1):
                target = k * a
                # Check if there's a subset of others of size k with sum = target
                # DP: dp[j][s] = can we pick j elements from first i elements of others summing to s
                # Use 1D DP: dp[j] = set of achievable sums with j elements
                # But sets are expensive. Use boolean arrays.
                
                # max possible sum with k elements from others (each <= 512) is k*512
                # min possible sum with k elements is k*1
                if target < k * min(others) or target > k * max(others):
                    continue
                
                # Use DP with bitsets for speed
                # dp[j] is a bitset (Python int) where bit s is set if sum s is achievable with j elements
                dp = [0] * (k + 1)
                dp[0] = 1  # sum 0 with 0 elements
                for val in others:
                    # Update in reverse order of j
                    for j in range(k, 0, -1):
                        dp[j] |= (dp[j-1] << val)
                # Check if bit target is set in dp[k]
                if (dp[k] >> target) & 1:
                    return False
        
        return True
    
    # Greedy with multiple orderings
    best_result = []
    
    orderings = [
        list(range(1, 513)),           # ascending
        list(range(512, 0, -1)),       # descending
        list(range(1, 513, 2)),        # odds only, ascending
        list(range(512, 0, -2)),       # odds only, descending
    ]
    
    # Add some heuristic orderings
    # Numbers that are "far" from others - e.g., large gaps
    # Try: numbers congruent to 1 mod some number
    for mod in [3, 4, 5]:
        orderings.append([x for x in range(1, 513) if x % mod == 1])
        orderings.append([x for x in range(512, 0, -1) if x % mod == 1])
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        selected = []
        for x in ordering:
            if time.time() > deadline:
                break
            candidate = selected + [x]
            if is_valid_set(candidate):
                selected = candidate
        
        if len(selected) > len(best_result):
            best_result = selected
    
    # If we haven't reached 48, try a more careful local search or different approach
    # But first, let's see what we have
    
    return best_result
