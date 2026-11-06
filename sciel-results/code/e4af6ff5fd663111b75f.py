import time
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 90s limit
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # We want to find a subset S such that sum_{i in S} 1/d_i < 1
    # Minimize 1 - sum.
    # This is equivalent to maximizing sum.
    
    # Strategy:
    # 1. Generate all possible sums of 2 distinct unit fractions that are < 1.
    #    Let T be such a sum. The deficit is 1 - T.
    #    If we can find a subset S containing the two denominators used for T 
    #    and some others such that sum(S) = T, then T is a valid solution.
    #    Actually, we just need sum(S) = T.
    #    Wait, if we fix the target T, we need to find a subset S of denominators 
    #    (excluding the ones that formed T? No, S can be any subset) 
    #    such that sum(S) = T.
    #    But T is formed by 1/a + 1/b. If we find a subset S summing to T, 
    #    it might not contain a and b. That's fine.
    #    However, if we just search for any subset summing to T, we might find 
    #    a subset that doesn't use the "large" denominators, which is fine.
    #    
    #    Better approach:
    #    The best solution has a very small deficit.
    #    The smallest possible deficit comes from the largest possible sum < 1.
    #    Let's generate candidate targets T in descending order of value (ascending deficit).
    #    For each T, check if T can be represented as a sum of distinct unit fractions 
    #    from the set {1/2, ..., 1/48}.
    #    
    #    To speed this up, we can use a "subset sum" style search for each target T.
    #    Since T is close to 1, the number of terms in the sum is likely small 
    #    (mostly 2 or 3 terms, maybe 4).
    #    
    #    Optimization:
    #    Precompute all sums of 1, 2, and 3 unit fractions.
    #    If a target T is in this set, we have found the optimal solution 
    #    (since we iterate T in descending order, the first hit is optimal).
    #    
    #    Why?
    #    The maximum sum < 1 is either a single fraction (max 1/2), 
    #    a sum of 2 fractions, or a sum of 3, etc.
    #    The largest possible sum < 1 using 2 fractions is close to 1.
    #    The largest possible sum < 1 using 3 fractions is even closer to 1?
    #    Not necessarily. 1/2 + 1/3 + 1/7 = 41/42 ~ 0.976.
    #    1/2 + 1/3 = 5/6 ~ 0.833.
    #    So 3-term sums can be larger than 2-term sums.
    #    
    #    Therefore, we should generate ALL possible sums of distinct unit fractions 
    #    that are < 1, sort them in descending order, and return the indices 
    #    of the first one found.
    #    
    #    How many such sums are there?
    #    Number of subsets is 2^47, which is too many.
    #    However, we only care about sums close to 1.
    #    
    #    Let's refine:
    #    Generate all sums of size k for k=2, 3, 4, ... up to some limit.
    #    Store them in a dictionary: sum_value -> list_of_indices.
    #    Then sort the keys (Fractions) in descending order.
    #    Return the indices for the largest key.
    #    
    #    Constraint: The sum must be STRICTLY BELOW 1.
    
    # Generate sums for subsets of size 2
    sums_dict = {}
    
    # Helper to add a sum to the dict
    def add_sum(val, indices):
        if val < 1:
            # If we already have this sum, we don't need to store multiple representations
            # unless one is "better" in some other metric, but the task only asks for indices.
            # Any valid set of indices is fine.
            if val not in sums_dict:
                sums_dict[val] = indices
    
    # Size 2
    for i in range(n):
        for j in range(i + 1, n):
            s = reciprocals[i] + reciprocals[j]
            add_sum(s, [i, j])
            
    # Size 3
    # To avoid explosion, we can prune. 
    # If reciprocals[i] + reciprocals[j] + reciprocals[k] >= 1, skip? 
    # No, we need strictly below 1.
    # We can iterate i, j, k.
    # Optimization: Sort indices by value descending.
    sorted_indices = sorted(range(n), key=lambda x: reciprocals[x], reverse=True)
    
    # Clear dict to avoid duplicates from different sizes if we just want the max sum.
    # Actually, we want the global max sum < 1.
    # So we can just keep the best sum found so far.
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Check size 1 (just in case, though 1/2 is the max single)
    for i in range(n):
        if reciprocals[i] > best_sum:
            best_sum = reciprocals[i]
            best_indices = [i]
            
    # Check size 2
    for i in range(n):
        for j in range(i + 1, n):
            s = reciprocals[i] + reciprocals[j]
            if s < 1 and s > best_sum:
                best_sum = s
                best_indices = [i, j]
                
    # Check size 3
    # Pruning: If reciprocals[i] + reciprocals[i+1] + reciprocals[i+2] <= best_sum, 
    # then no further triple starting with i can beat best_sum? 
    # Not exactly, but if the max possible sum for remaining is small, we can stop.
    
    # To make it faster, iterate i, j, k in sorted order.
    # Since we want to maximize sum, we start with largest fractions.
    
    # Let's use a more efficient search for size 3, 4, etc.
    # But wait, the current program's score is ~33000, which corresponds to a deficit of ~1/33000.
    # 1/33000 is very small.
    # 1/2 + 1/3 + 1/7 = 41/42. Deficit 1/42. Score 42.
    # To get score 33000, we need deficit ~1/33000.
    # This implies a very specific combination, likely involving many terms or very specific denominators.
    # Example: 1/2 + 1/3 + 1/7 + 1/43 ...
    
    # The "Generate all sums" approach for size 4 is 48^4 / 24 ~ 1.8 million. That's fine.
    # Size 5 is 48^5 / 120 ~ 22 million. Might be too slow in Python with Fractions.
    
    # Let's try to find the optimal solution by searching for sums close to 1.
    # We can iterate through possible deficits.
    # Deficit D = 1 - S.
    # We want to minimize D.
    
    # Alternative Strategy:
    # The best solution is likely found by a greedy-like approach but with local search or 
    # a more targeted DFS that uses tight bounds.
    # The current DFS uses `ub = current_sum + suffix_sums[idx]`.
    # `suffix_sums[idx]` is the sum of ALL remaining elements.
    # This is a very loose bound.
    # A much tighter bound: `current_sum + sum of largest k remaining elements that fit`.
    
    # Let's implement a tighter Branch and Bound.
    # Sort denominators descending.
    # Precompute suffix sums.
    # In DFS(idx, current_sum, current_set):
    #   if time > limit: return
    #   if current_sum > best_sum: update best
    #   if idx == n: return
    #   
    #   # Pruning
    #   # Max possible addition is bounded by the capacity (1 - current_sum)
    #   # and the sum of remaining items.
    #   # But we can do better: 
    #   # If current_sum + suffix_sums[idx] <= best_sum: return
    #   
    #   # Try adding item idx
    #   if current_sum + reciprocals[idx] < 1:
    #       dfs(idx + 1, current_sum + reciprocals[idx], current_set | {idx})
    #       
    #   # Try skipping item idx
    #   dfs(idx + 1, current_sum, current_set)
    
    # The issue with the current program is that it explores "Skip" first? 
    # No, it explores "Skip" then "Take".
    # If we explore "Take" first, we might find a good solution faster, 
    # allowing the bound to prune more "Skip" branches.
    
    # Let's try "Take" first.
    # Also, use integer arithmetic where possible or optimize Fraction usage.
    # Fractions are slow.
    
    # Can we use floats for search and Fractions for verification?
    # Risk: Floating point errors might lead to invalid solutions (sum >= 1).
    # We can use a small epsilon.
    # Or, we can use integers: 
    # Sum of 1/d_i < 1  <=>  Sum(LCM/d_i) < LCM.
    # LCM of 1..48 is huge.
    
    # Let's stick to Fractions but optimize the search order and bounds.
    
    # Revised Strategy:
    # 1. Sort indices by value descending.
    # 2. Precompute suffix sums (Fraction).
    # 3. DFS with "Take" first.
    # 4. Tighter bound: 
    #    The maximum sum we can add is at most `1 - current_sum`.
    #    So if `current_sum + min(suffix_sums[idx], 1 - current_sum) <= best_sum`, prune.
    #    Actually, `current_sum + suffix_sums[idx]` is the absolute max sum of remaining.
    #    If that max sum is <= best_sum, prune.
    #    This is what the current program does.
    
    # Why is the current program slow?
    # 47 items. 2^47 is huge.
    # The bound `suffix_sums[idx]` is the sum of ALL remaining.
    # For large idx, suffix_sums is small.
    # For small idx, suffix_sums is large (~1.5).
    # So pruning is only effective when we are deep in the tree or have a very good best_sum.
    
    # Improvement:
    # Use a "Knapsack" style bound.
    # If we sort by value/weight (here value=1/d, weight=1/d, ratio=1), 
    # the fractional knapsack bound is tight.
    # Since all ratios are 1, the fractional bound is just the capacity.
    # So `ub = min(1, current_sum + suffix_sums[idx])`.
    # If `ub <= best_sum`, prune.
    # This is slightly tighter than `current_sum + suffix_sums[idx]` if the sum exceeds 1.
    # But since we are constrained by < 1, the max possible sum is 1 (exclusive).
    # So `ub = min(1, current_sum + suffix_sums[idx])`.
    # If `best_sum` is very close to 1, this prunes a lot.
    
    # Let's implement this with "Take" first.
    
    sorted_indices = sorted(range(n), key=lambda i: reciprocals[i], reverse=True)
    
    # Precompute suffix sums
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + reciprocals[sorted_indices[i]]
        
    best_sum = Fraction(0)
    best_set = set()
    
    # Initial greedy to set a good best_sum
    curr_sum = Fraction(0)
    curr_set = set()
    for i in sorted_indices:
        if curr_sum + reciprocals[i] < 1:
            curr_sum += reciprocals[i]
            curr_set.add(i)
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_set = set(curr_set)
        
    def dfs(idx, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if time.time() - start_time > timeout:
            return
            
        # Pruning
        # Max possible sum we can reach is min(1, current_sum + suffix_sums[idx])
        # If this is <= best_sum, we cannot improve.
        max_possible = current_sum + suffix_sums[idx]
        if max_possible >= 1:
            # The theoretical max is 1 (exclusive).
            # If best_sum is very close to 1, we can't beat it unless we find exactly 1 (impossible) 
            # or a sum closer to 1 than best_sum.
            # Since we can't reach 1, the next best thing is the largest sum < 1.
            # If best_sum is, say, 1 - 1/1000, and max_possible is 1.5, 
            # we don't know if we can find 1 - 1/1001.
            # So we can't prune just because max_possible >= 1.
            # We can only prune if max_possible <= best_sum.
            pass
            
        if max_possible <= best_sum:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
            
        i = sorted_indices[idx]
        
        # Option 1: Take item idx
        new_sum = current_sum + reciprocals[i]
        if new_sum < 1:
            current_set.add(i)
            dfs(idx + 1, new_sum, current_set)
            current_set.remove(i)
            
        # Option 2: Skip item idx
        dfs(idx + 1, current_sum, current_set)
        
    dfs(0, Fraction(0), set())
    
    if not best_set:
        # Fallback
        best_set = {0} # 1/2 is always valid
        
    return list(best_set)
