import numpy as np
import time

def construct():
    n = 5040
    target = 4979
    
    # 1. Calculate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort descending for reconstruction efficiency (greedy-like pruning)
    divisors.sort(reverse=True)
    
    # 2. DP to find minimum count
    # Using NumPy for speed
    INF = 10000
    dp = np.full(target + 1, INF, dtype=np.int32)
    dp[0] = 0
    
    start_time = time.time()
    deadline = 50.0
    
    for d in divisors:
        if time.time() - start_time > deadline:
            break
        # Vectorized update: dp[s] = min(dp[s], dp[s-d] + 1)
        # We need to be careful with the slice to avoid using updated values in the same iteration
        # NumPy slice assignment is done on a copy of the source slice, so this is safe for 0/1 knapsack
        # dp[d:] refers to the destination
        # dp[:-d] refers to the source (previous state)
        # However, we must ensure we don't go out of bounds.
        if d <= target:
            dp[d:] = np.minimum(dp[d:], dp[:-d] + 1)
            
    min_count = dp[target]
    
    if min_count >= INF:
        # Fallback: Greedy (should not happen)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    # 3. Reconstruct the solution
    # We know the minimum count is min_count.
    # We search for a subset of size min_count that sums to target.
    # Use backtracking with pruning.
    
    # Precompute suffix sums of sorted divisors for pruning
    # suffix_sum[i] = sum of divisors from i to end
    # Actually, since we want to pick exactly min_count items, 
    # we can prune if the sum of the largest min_count available items is < remaining_sum
    # Or if the sum of the smallest min_count available items is > remaining_sum
    
    # Let's use a simple recursive search with memoization or pruning
    # Since min_count is likely small (e.g., < 10), and divisors are 60,
    # the search space is manageable.
    
    def find_subset(remaining_sum, remaining_count, start_idx):
        # If we need 0 more items, check if sum is 0
        if remaining_count == 0:
            return [] if remaining_sum == 0 else None
        
        # Pruning:
        # If remaining_sum is 0, we need 0 more items (handled above)
        if remaining_sum <= 0:
            return None if remaining_sum != 0 else [] # Should not happen with positive divisors
        
        # If we don't have enough divisors left
        if len(divisors) - start_idx < remaining_count:
            return None
            
        # Pruning: Max possible sum with remaining_count items from start_idx
        # The largest items are at the beginning.
        # Sum of largest 'remaining_count' items available:
        max_possible_sum = sum(divisors[start_idx : start_idx + remaining_count])
        if max_possible_sum < remaining_sum:
            return None
            
        # Pruning: Min possible sum with remaining_count items from start_idx
        # The smallest items are at the end.
        # We need to pick 'remaining_count' items from divisors[start_idx:]
        # The minimum sum is the sum of the last 'remaining_count' items in that range.
        min_possible_sum = sum(divisors[len(divisors) - remaining_count :]) # Wait, this is global min.
        # Better: sum of the smallest 'remaining_count' items in divisors[start_idx:]
        # Since divisors are sorted descending, the smallest are at the end.
        # Indices: start_idx to end.
        # Smallest 'remaining_count' items are at indices: len(divisors)-remaining_count to len(divisors)-1
        # But we must start from start_idx.
        # If start_idx > len(divisors) - remaining_count, we can't pick the global smallest.
        # We pick the last 'remaining_count' items from the available range.
        available = divisors[start_idx:]
        if len(available) < remaining_count:
            return None
        min_possible_sum = sum(available[-remaining_count:])
        if min_possible_sum > remaining_sum:
            return None

        # Try each divisor in the available range
        # To find a solution quickly, try larger divisors first
        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            if d > remaining_sum:
                continue # Since sorted desc, all subsequent are smaller? No, sorted desc, so subsequent are smaller. 
                # Wait, if d > remaining_sum, we can skip this one, but smaller ones might fit.
                # So we cannot break, we must continue.
                
            # If we pick d, we need to pick remaining_count-1 items from i+1
            sub_sol = find_subset(remaining_sum - d, remaining_count - 1, i + 1)
            if sub_sol is not None:
                return [d] + sub_sol
                
        return None

    # Start search
    solution = find_subset(target, min_count, 0)
    
    if solution is None:
        # Fallback if search fails (should not happen if DP is correct)
        # Return a greedy solution
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    return solution
