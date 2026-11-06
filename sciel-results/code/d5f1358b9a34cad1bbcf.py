"""Improved program with bitset-based subset sum for fast mean checking."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    n = 512
    target_size = 48
    
    def _build_dp(arr):
        """Build DP table: dp[k] is bitset of all subset sums of size k."""
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        return dp
    
    def _can_form_sum_dp(dp, target_sum, min_size, max_size):
        """Check using precomputed DP if subset of size in [min_size, max_size] sums to target_sum."""
        if target_sum < 0:
            return False
        m = len(dp) - 1
        max_size = min(max_size, m)
        if max_size < min_size:
            return False
        for k in range(min_size, max_size + 1):
            if (dp[k] >> target_sum) & 1:
                return True
        return False
    
    def _is_valid_incremental(existing, x):
        """Check if adding x to existing set keeps it valid. Uses DP for speed."""
        if not existing:
            return True
        m = len(existing)
        total_sum = sum(existing)
        
        # Precompute DP for existing
        dp = _build_dp(existing)
        
        # Check 1: x is not the mean of any subset of existing of size >= 2
        for k in range(2, m + 1):
            target = k * x
            if target > total_sum:
                break
            if (dp[k] >> target) & 1:
                return False
        
        # Check 2: no element a in existing is the mean of a subset of existing∪{x} of size >= 2 that includes x
        # Such a subset has x plus t elements from existing, total size t+1 >= 2, so t >= 1
        # Mean = a means: (sum of t elements from existing + x) / (t+1) = a
        # => sum of t elements = (t+1)*a - x
        for a in existing:
            for t in range(1, m + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target > total_sum:
                    break
                if (dp[t] >> target) & 1:
                    return False
        return True
    
    def _is_valid_full(selected):
        """Full validation of a complete set."""
        m = len(selected)
        if m < 2:
            return True
        dp = _build_dp(selected)
        for i, a in enumerate(selected):
            # Check if a is the mean of any subset of size >= 2 not containing a
            others = [selected[j] for j in range(m) if j != i]
            if not others:
                continue
            dp_others = _build_dp(others)
            for k in range(2, len(others) + 1):
                target = k * a
                if target > sum(others):
                    break
                if (dp_others[k] >> target) & 1:
                    return False
        return True
    
    def _greedy_from_order(order, selected_start=None):
        """Try to build a valid set by greedily adding elements in the given order."""
        if selected_start is None:
            selected = []
        else:
            selected = list(selected_start)
        for x in order:
            if x in selected:
                continue
            if _is_valid_incremental(selected, x):
                selected.append(x)
                if len(selected) >= target_size:
                    break
        return selected
    
    def _local_search(selected, max_iterations=30):
        """Try to improve the set by swapping elements."""
        selected = list(selected)
        for _ in range(max_iterations):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for i in range(len(selected)):
                if time.time() > deadline - 3.0:
                    break
                candidate_remove = selected[i]
                remaining = [s for j, s in enumerate(selected) if j != i]
                for x in range(1, n + 1):
                    if x in remaining or x == candidate_remove:
                        continue
                    if _is_valid_incremental(remaining, x):
                        new_selected = remaining + [x]
                        if len(new_selected) >= len(selected):
                            selected = new_selected
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return selected
    
    # Strategy 1: Powers of 2 and their neighbors - these have few subset-sum representations
    # Start with powers of 2: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
    # Then try to add other numbers that don't create subset mean issues
    powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    best = _greedy_from_order(powers + list(range(3, n + 1)))
    
    if len(best) >= target_size:
        return best
    
    # Strategy 2: Greedy from natural order
    order = list(range(1, n + 1))
    result = _greedy_from_order(order)
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best
    
    # Strategy 3: Greedy from descending order
    order_desc = list(range(n, 0, -1))
    result = _greedy_from_order(order_desc)
    if len(result) > len(best):
        best = result
        if len(best) >= target_size:
            return best
    
    # Strategy 4: Try different random orderings
    for trial in range(100):
        if time.time() > deadline - 15.0:
            break
        random.seed(trial)
        order = list(range(1, n + 1))
        random.shuffle(order)
        result = _greedy_from_order(order)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best
    
    # Strategy 5: Local search on best found
    if len(best) < target_size:
        improved = _local_search(best, max_iterations=20)
        if len(improved) > len(best):
            best = improved
            if len(best) >= target_size:
                return best
    
    # Strategy 6: Try starting with large numbers and working down
    # Large numbers are less likely to be means of subsets of smaller numbers
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        random.seed(2000 + trial)
        # Mix of large and small numbers
        large_nums = list(range(n, n // 2, -1))
        small_nums = list(range(1, n // 2 + 1))
        random.shuffle(large_nums)
        random.shuffle(small_nums)
        order = large_nums[:len(large_nums)//2] + small_nums[:len(small_nums)//2]
        order += large_nums[len(large_nums)//2:] + small_nums[len(small_nums)//2:]
        random.shuffle(order)
        result = _greedy_from_order(order)
        if len(result) > len(best):
            best = result
            if len(best) >= target_size:
                return best
    
    # Strategy 7: Try to construct using numbers that are "sparse" in their subset sums
    # Numbers that are sums of distinct powers of 2 (all numbers) but with specific structure
    # Try numbers congruent to certain values mod small numbers
    for mod in [3, 4, 5, 6, 7]:
        if time.time() > deadline - 3.0:
            break
        for rem in range(mod):
            order = [x for x in range(1, n + 1) if x % mod == rem]
            result = _greedy_from_order(order)
            if len(result) > len(best):
                best = result
                if len(best) >= target_size:
                    return best
    
    # Strategy 8: Final local search with more iterations
    if len(best) < target_size and time.time() < deadline - 2.0:
        improved = _local_search(best, max_iterations=50)
        if len(improved) > len(best):
            best = improved
    
    return best
# EVOLVE-BLOCK-END
