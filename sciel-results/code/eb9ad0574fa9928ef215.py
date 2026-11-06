"""Combined program: Incremental DP maintenance + Multi-strategy search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    n = 512
    target_size = 48
    
    # Precompute powers of 2 for potential use or just as a fast check
    # The core logic relies on bitset DP
    
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

    def _is_valid_incremental(dp, m, total_sum, existing, x):
        """
        Check if adding x to existing (with precomputed dp, m, total_sum) keeps it valid.
        dp[k] is bitset of subset sums of size k from 'existing'.
        m is len(existing).
        """
        if not existing:
            return True
        
        # Check 1: x is not the mean of any subset of existing of size >= 2
        # Mean = x => Sum = k * x
        for k in range(2, m + 1):
            target = k * x
            if target > total_sum:
                break
            if (dp[k] >> target) & 1:
                return False
        
        # Check 2: no element a in existing is the mean of a subset of existing∪{x} 
        # of size >= 2 that includes x.
        # Subset = {x} U T, where T is subset of existing, |T| = t.
        # Size = t + 1 >= 2 => t >= 1.
        # Mean = a => (x + sum(T)) / (t + 1) = a => sum(T) = (t + 1)*a - x
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

    def _update_dp(dp, m, x):
        """Update dp to include x. Returns new dp list."""
        # We need to update from high k to low k to avoid using x multiple times
        # dp is currently for size m. We need dp for size m+1.
        # New dp size is m+2.
        new_dp = [0] * (m + 2)
        # Copy existing
        for k in range(m + 1):
            new_dp[k] = dp[k]
        # Update with x
        # For k from m down to 1: new_dp[k] |= old_dp[k-1] << x
        # Note: new_dp[m+1] will be dp[m] << x
        for k in range(m, 0, -1):
            new_dp[k] |= (dp[k-1] << x)
        return new_dp

    def _greedy_with_dp(order, selected_start=None, dp_start=None, m_start=None, total_sum_start=None):
        """
        Greedy construction using incremental DP.
        """
        if selected_start is None:
            selected = []
            dp = [1]
            m = 0
            total_sum = 0
        else:
            selected = list(selected_start)
            dp = dp_start if dp_start else _build_dp(selected)
            m = m_start if m_start is not None else len(selected)
            total_sum = total_sum_start if total_sum_start is not None else sum(selected)
            
        for x in order:
            if time.time() > deadline - 2.0:
                break
            if x in selected:
                continue
            
            if _is_valid_incremental(dp, m, total_sum, selected, x):
                # Valid, add it
                selected.append(x)
                dp = _update_dp(dp, m, x)
                m += 1
                total_sum += x
                if len(selected) >= target_size:
                    break
        
        return selected, dp, m, total_sum

    def _local_search(selected, dp, m, total_sum, max_iterations=30):
        """
        Try to improve the set by swapping elements or finding additions.
        Since we want to maximize size, we primarily look for additions.
        If we are stuck at size K < 48, we try to swap an element to make room for more.
        """
        selected = list(selected)
        # We need to be able to remove an element from DP? 
        # Removing from DP is hard (inversion of OR is not possible).
        # So for local search, we might just rebuild DP after a swap.
        
        for _ in range(max_iterations):
            if time.time() > deadline - 3.0:
                break
            improved = False
            # Try to find an element x not in selected that can be added
            # This is what greedy did. If greedy failed, maybe a different order helps?
            # Or maybe we need to swap.
            
            # Strategy: Pick a random element to remove, rebuild DP, then try to add 2 elements?
            # Or just try to add 1 element to the remaining set.
            # If we remove 1 and add 1, size stays same.
            # If we remove 1 and add 2, size increases by 1.
            
            # Let's try simple swap: Remove i, try to add x.
            # This doesn't increase size.
            
            # To increase size, we must find a "hole" that greedy missed.
            # This is hard. Let's stick to greedy with better orders.
            # Local search might be better used to refine the set for *validity* if we were doing exact cover, 
            # but here we just want max size.
            
            # Let's try a "kick" move: Remove a random element, then run greedy fill from scratch 
            # on the remaining set + the removed element + others, prioritizing the removed one?
            # No, that's complex.
            
            # Alternative: Randomly shuffle the order of *unselected* candidates and retry greedy fill 
            # starting from the current state?
            # This is essentially what multiple random orders do.
            
            # Let's just return the current best. The multi-order greedy is the main driver.
            pass
            
        return selected

    # Strategy 1: Greedy from descending order (Large numbers first)
    # This is often the best strategy for this specific problem because large numbers
    # are less likely to be the mean of smaller numbers.
    order_desc = list(range(n, 0, -1))
    best, best_dp, best_m, best_sum = _greedy_with_dp(order_desc)
    
    if len(best) >= target_size:
        return best

    # Strategy 2: Greedy from natural order
    order_nat = list(range(1, n + 1))
    res, _, _, _ = _greedy_with_dp(order_nat)
    if len(res) > len(best):
        best, best_dp, best_m, best_sum = res, None, None, None # DP needs rebuild if we switch context, but we only use best for size
        if len(best) >= target_size:
            return best

    # Strategy 3: Greedy from powers of 2 start, then fill
    powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Ensure powers are valid subset
    # We can just prepend them to a shuffled rest
    rest = [x for x in range(1, n + 1) if x not in powers]
    random.seed(42)
    random.shuffle(rest)
    order_powers = powers + rest
    res, _, _, _ = _greedy_with_dp(order_powers)
    if len(res) > len(best):
        best = res
        if len(best) >= target_size:
            return best

    # Strategy 4: Multiple random orders
    for trial in range(100):
        if time.time() > deadline - 10.0:
            break
        random.seed(1000 + trial)
        order = list(range(1, n + 1))
        random.shuffle(order)
        # Bias towards larger numbers?
        # Sort by random key but favor larger?
        # Just shuffle for now.
        res, _, _, _ = _greedy_with_dp(order)
        if len(res) > len(best):
            best = res
            if len(best) >= target_size:
                return best

    # Strategy 5: Try to fill gaps from the best set found
    # If we have a set of size K < 48, try to add more elements.
    # We need the DP of the best set.
    if len(best) < target_size:
        # Rebuild DP for best
        dp_best = _build_dp(best)
        m_best = len(best)
        sum_best = sum(best)
        
        # Try adding any remaining number
        remaining = [x for x in range(1, n + 1) if x not in best]
        # Sort remaining by "safety" (e.g., large numbers first)
        remaining.sort(reverse=True)
        
        for x in remaining:
            if time.time() > deadline - 5.0:
                break
            if _is_valid_incremental(dp_best, m_best, sum_best, best, x):
                best.append(x)
                dp_best = _update_dp(dp_best, m_best, x)
                m_best += 1
                sum_best += x
                if len(best) >= target_size:
                    return best

    return best
# EVOLVE-BLOCK-END
