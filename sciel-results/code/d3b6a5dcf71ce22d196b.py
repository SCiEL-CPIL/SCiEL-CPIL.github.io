"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    N = 100
    target = 200
    
    def find_subset_sum_dp(elements):
        """Find a subset of elements summing to target. Returns (subset, all_witnesses) or (None, None).
        all_witnesses: list of all distinct subsets found (up to a limit)."""
        n = len(elements)
        if n == 0:
            return None, []
        # DP: dp[s] = set of indices used, but that's too expensive.
        # Instead: dp[s] = list of (prev_sum, elem_idx) to allow multiple paths.
        # For efficiency, we track reachable sums and parent pointers.
        reachable = {0}
        parent = {}  # sum -> (prev_sum, elem_idx)
        for i, x in enumerate(elements):
            new_sums = []
            for s in list(reachable):
                ns = s + x
                if ns <= target:
                    if ns not in parent:
                        parent[ns] = (s, i)
                    new_sums.append(ns)
            reachable.update(new_sums)
            if target in reachable:
                break
        if target not in reachable:
            return None, []
        # Reconstruct one subset
        subset = []
        cur = target
        while cur > 0:
            prev_s, i = parent[cur]
            subset.append(elements[i])
            cur = prev_s
        return subset, []
    
    def count_witnesses(elements, limit=5000):
        """Count how many distinct subsets of elements sum to target, up to limit.
        Returns (count, list_of_witnesses) where witnesses are lists of element values."""
        n = len(elements)
        if n == 0:
            return 0, []
        # Use DP with bitsets for efficiency
        # dp[s] = number of ways (capped), and store some witnesses
        dp = [0] * (target + 1)
        dp[0] = 1
        witnesses = []
        # Track parent info for reconstruction of a few witnesses
        parent_sum = [-1] * (target + 1)
        parent_elem = [-1] * (target + 1)
        
        for i, x in enumerate(elements):
            for s in range(target - x, -1, -1):
                if dp[s] > 0:
                    ns = s + x
                    if dp[ns] == 0:
                        parent_sum[ns] = s
                        parent_elem[ns] = i
                    dp[ns] = min(dp[ns] + dp[s], limit + 1)
        
        count = dp[target]
        if count == 0:
            return 0, []
        
        # Reconstruct up to 'limit' witnesses via DFS with memoization
        # For small counts, this is feasible
        if count <= limit:
            # Backtrack to find all witnesses
            # This could be expensive, so we limit
            def backtrack(s, current_subset):
                if len(witnesses) >= limit:
                    return
                if s == 0:
                    witnesses.append(list(current_subset))
                    return
                if s < 0:
                    return
                # Find which element was last added
                # We need to try all possible last elements
                for i in range(n - 1, -1, -1):
                    x = elements[i]
                    if x <= s and parent_sum[s] == s - x and parent_elem[s] == i:
                        current_subset.append(x)
                        backtrack(s - x, current_subset)
                        current_subset.pop()
                        if len(witnesses) >= limit:
                            return
                    # Also check if this element could have been used in a different way
                    # Actually the DP only stores one parent, so we can only follow that path
                    # For a proper count of all subsets, we'd need more info.
                    # Let's just return the count and one witness for now.
                    break
            
            backtrack(target, [])
        
        return min(count, limit), witnesses[:limit]
    
    def greedy_largest_first():
        selected = []
        reachable = {0}
        for x in range(N, 0, -1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected, reachable
    
    def greedy_smallest_first():
        selected = []
        reachable = {0}
        for x in range(1, N + 1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected, reachable
    
    # Initialize with greedy solutions
    selected, reachable = greedy_largest_first()
    best_selected = list(selected)
    best_size = len(best_selected)
    
    sel2, reach2 = greedy_smallest_first()
    if len(sel2) > best_size:
        best_selected = list(sel2)
        best_size = len(best_selected)
    
    if time.time() > deadline - 2:
        return best_selected
    
    # Main approach: start with full set, iteratively remove elements using
    # a degree-based heuristic (remove element that participates in most witness subsets)
    
    def solve_with_degree_heuristic(seed=42):
        """Iteratively remove elements, choosing the one in the most witness subsets."""
        rng = random.Random(seed)
        full_set = list(range(1, N + 1))
        remaining = list(full_set)
        removed = set()
        
        max_iterations = 50
        for it in range(max_iterations):
            if time.time() > deadline - 3:
                break
            if not remaining:
                break
            
            subset, witnesses = find_subset_sum_dp(remaining)
            if subset is None:
                break
            
            # Count how many times each element in the subset appears across witnesses
            # If we don't have many witnesses, just use the current subset
            if len(witnesses) > 1:
                elem_count = {}
                for w in witnesses:
                    for e in w:
                        elem_count[e] = elem_count.get(e, 0) + 1
                # Pick element with highest count; tie-break randomly
                max_count = max(elem_count.values())
                candidates = [e for e in subset if elem_count.get(e, 0) == max_count]
                to_remove = rng.choice(candidates)
            else:
                # Only one witness; pick element that appears in most other potential sums
                # Heuristic: remove the element whose removal breaks the most "near-miss" sums
                # Simpler: remove the largest, or random
                to_remove = rng.choice(subset)
            
            removed.add(to_remove)
            remaining = [x for x in remaining if x != to_remove]
        
        # Local search: try to add back removed elements
        for elem in sorted(removed, reverse=True):
            if time.time() > deadline - 3:
                break
            test_remaining = remaining + [elem]
            subset_test, _ = find_subset_sum_dp(test_remaining)
            if subset_test is None:
                remaining.append(elem)
                removed.discard(elem)
        
        return remaining
    
    # Run degree-based heuristic with several seeds
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        result = solve_with_degree_heuristic(seed)
        if len(result) > best_size:
            best_selected = list(result)
            best_size = len(best_selected)
    
    if time.time() > deadline - 3:
        return best_selected
    
    # Random restarts with random removal
    rng = random.Random(123)
    for restart in range(100):
        if time.time() > deadline - 3:
            break
        removed_set = set()
        remaining = list(range(1, N + 1))
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            subset, witnesses = find_subset_sum_dp(remaining)
            if subset is None:
                break
            if len(witnesses) > 1:
                elem_count = {}
                for w in witnesses:
                    for e in w:
                        elem_count[e] = elem_count.get(e, 0) + 1
                max_count = max(elem_count.values())
                candidates = [e for e in subset if elem_count.get(e, 0) >= max_count * 0.8]
                to_remove = rng.choice(candidates)
            else:
                to_remove = rng.choice(subset)
            removed_set.add(to_remove)
            remaining = [x for x in remaining if x != to_remove]
        
        # Local search: try to add back
        for elem in sorted(removed_set, reverse=True):
            if time.time() > deadline - 3:
                break
            test_remaining = remaining + [elem]
            subset_test, _ = find_subset_sum_dp(test_remaining)
            if subset_test is None:
                remaining.append(elem)
                removed_set.discard(elem)
        
        if len(remaining) > best_size:
            best_selected = list(remaining)
            best_size = len(best_selected)
    
    return best_selected
# EVOLVE-BLOCK-END
