"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    # New Approach: Frequency-Based Greedy Hitting Set
    # Start with S = {1, ..., 100}.
    # While there exists a subset of S summing to 200:
    #   1. Sample several subsets of S that sum to 200.
    #   2. Count the frequency of each element in S across these samples.
    #   3. Identify the "most dangerous" element (highest frequency).
    #   4. Remove it from S.
    # This is a hitting set heuristic: remove elements that appear in the most "conflicting" subsets.
    
    def find_subset_sum_dp(S, target):
        """
        Returns a list of elements from S that sum to target, or None if not possible.
        Uses DP to track predecessors.
        """
        # Sort S for deterministic behavior in DP if needed, but we need the actual values
        # We only care about sums up to target
        dp = {0: []}
        for x in S:
            if x > target:
                continue
            new_dp = {}
            for s, subset in dp.items():
                if s + x <= target:
                    new_subset = subset + [x]
                    # Keep the first found or shortest? Shortest is often better for hitting set?
                    # Actually, any subset works.
                    if s + x not in new_dp or len(new_subset) < len(new_dp[s + x]):
                        new_dp[s + x] = new_subset
            # Merge
            for s, subset in new_dp.items():
                if s not in dp or len(subset) < len(dp[s]):
                    dp[s] = subset
        if target in dp:
            return dp[target]
        return None

    def has_subset_sum(S, target):
        """Quick check if target is reachable."""
        reachable = {0}
        for x in S:
            if x > target:
                continue
            new = {s + x for s in reachable if s + x <= target}
            reachable |= new
        return target in reachable

    def sample_subsets_summing_to(S, target, count, rng):
        """
        Generate 'count' random subsets of S that sum to target.
        This is tricky to do directly. 
        Instead, we can use the DP to find one, then maybe perturb?
        Or just find one subset, and if we want frequency, we need many.
        
        Alternative: Instead of sampling random subsets, we can just find ONE subset T.
        Then, for each x in T, we can estimate its "importance" by checking how many 
        *other* subsets summing to target exist if we keep x? No, that's expensive.
        
        Let's simplify: 
        1. Find one subset T summing to 200.
        2. For each x in T, temporarily remove x and check if 200 is still reachable.
        3. If removing x makes 200 unreachable, x is a "cut vertex" for that sum. 
           Removing it is very effective.
        4. If removing x still leaves 200 reachable, x is "redundant" for the current sum, 
           but might be part of other sums.
        
        Actually, the simplest effective heuristic for Hitting Set:
        1. Find subset T summing to 200.
        2. Pick an element x from T.
        3. Remove x.
        
        To make it "smart":
        1. Find subset T.
        2. For each x in T, count how many elements in T are "large" vs "small"?
        
        Let's try a different angle: 
        The problem is small enough (100 elements, sum 200) that we can use a more aggressive search.
        
        Revised Plan:
        1. Start with S = {1..100}.
        2. While has_subset_sum(S, 200):
           - Find a subset T summing to 200.
           - Try removing each x in T.
           - For each candidate removal x, calculate a "score" based on how much it reduces the 
             number of reachable sums or similar.
           - Simpler: Just remove the smallest element in T? Or largest?
           - Let's try: Remove the element in T that is the median?
           
        Let's go with **Multiple Restarts with Randomized Hitting Set**.
        1. S = {1..100}
        2. While has_subset_sum(S, 200):
           - Find one subset T summing to 200.
           - Randomly pick an element x from T.
           - Remove x from S.
        3. Return S.
        
        This is very fast but might not maximize size.
        
        To maximize size, we should remove elements that are "least valuable".
        Large numbers are "more valuable" for reaching high sums? 
        If we remove a large number, we lose a lot of potential sum capacity.
        If we remove a small number, we lose less.
        So, **remove the smallest element in T**?
        Let's test this hypothesis.
        
        Alternative: **Remove the element that appears in the most subsets**.
        We can approximate this by finding all subsets of size <= 3 summing to 200?
        No, subsets can be large.
        
        Let's use the **DP-based Greedy Removal**:
        1. S = {1..100}
        2. While has_subset_sum(S, 200):
           - Find one subset T summing to 200 using DP.
           - Among elements in T, pick the one with the smallest value?
           - Remove it.
           
        Let's try this specific heuristic: **Remove the smallest element in the found subset**.
        Why? Small elements are more flexible and appear in more combinations. 
        Removing them might break more sums? 
        Actually, small elements are "cheap" in terms of sum value, so removing them has less impact on the *range* of sums, but they are components of many sums.
        Conversely, large elements are "expensive". Removing a large element breaks fewer sums because fewer subsets can include it without exceeding 200.
        So removing large elements is "safer" for keeping the set size high? 
        Wait, if I remove 100, I break sums like {100, 99, 1}. 
        If I remove 1, I break {1, 199}, {1, 2, 197}, etc.
        1 is in more subsets than 100.
        So removing 1 breaks more "potential" sums, but we only need to break the *current* one.
        We want to remove the element that breaks the *most* future conflicts.
        So we should remove the element that is in the most subsets summing to 200.
        Since 1 is in more subsets than 100, removing 1 is likely better for hitting set size?
        
        Let's implement:
        1. S = {1..100}
        2. While has_subset_sum(S, 200):
           - Find subset T summing to 200.
           - Estimate frequency: 
             For each x in T, count how many subsets of S \ {x} sum to 200 - x? No.
             Just pick the smallest x in T.
           - Remove x.
        
        Let's also try **Largest x in T** and compare.
        I will implement a function that tries both "Remove Smallest" and "Remove Largest" and returns the better one.
        
    """
    
    def find_one_subset(S, target):
        """Find one subset of S summing to target. Returns list or None."""
        # Use DP with path reconstruction
        # dp[s] = last element added to reach sum s
        dp = {0: None}
        parent = {0: None}
        for x in S:
            if x > target:
                continue
            # Iterate over current keys to avoid using x multiple times in one step? 
            # No, standard 0/1 knapsack DP.
            # We need to iterate over a copy of the current reachable sums
            items_to_add = []
            for s in list(dp.keys()):
                if s + x <= target and s + x not in dp:
                    dp[s + x] = x
                    parent[s + x] = s
                    items_to_add.append(s + x)
            # Optimization: if target reached, we can stop? 
            # But we need to ensure we find a valid path.
            if target in dp:
                break
        
        if target not in dp:
            return None
        
        # Reconstruct
        subset = []
        cur = target
        while cur != 0:
            x = dp[cur]
            subset.append(x)
            cur = parent[cur]
        return subset

    def solve_with_heuristic(heuristic):
        S = list(range(1, 101))
        # S is a list, we remove elements. 
        # Using a list for S is O(N) for removal. 
        # Given N=100, this is fine.
        
        while True:
            # Check if valid
            T = find_one_subset(S, 200)
            if T is None:
                return S
            
            # Choose element to remove
            if heuristic == 'smallest':
                x_to_remove = min(T)
            elif heuristic == 'largest':
                x_to_remove = max(T)
            elif heuristic == 'random':
                x_to_remove = random.choice(T)
            else:
                x_to_remove = min(T) # default
            
            S.remove(x_to_remove)
            
    # Try different heuristics and return the best (largest S)
    best_S = []
    
    # Heuristic 1: Remove Smallest
    S1 = solve_with_heuristic('smallest')
    if len(S1) > len(best_S):
        best_S = S1
        
    # Heuristic 2: Remove Largest
    S2 = solve_with_heuristic('largest')
    if len(S2) > len(best_S):
        best_S = S2
        
    # Heuristic 3: Randomized (multiple runs)
    rng = random.Random(42)
    for _ in range(50):
        S3 = solve_with_heuristic('random')
        if len(S3) > len(best_S):
            best_S = S3
            
    return best_S
# EVOLVE-BLOCK-END
