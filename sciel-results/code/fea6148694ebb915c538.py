"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin from 90s limit

    best_set = []
    best_len = 0

    # Helper to check if x is a mean of a subset of size k
    # dp[k] is a bitmask of achievable sums with exactly k elements
    def is_mean_of_subset(dp, x, k):
        target = k * x
        if target < 0 or target > 24576:
            return False
        return (dp[k] >> target) & 1

    def try_greedy_v2(order):
        selected = []
        n = 0
        dp_all = [1] # dp_all[0] = 1
        dp_without = {}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= 48:
                break
            
            valid = True
            
            # Check 1
            for k in range(2, n + 1):
                target = k * x
                if target > 24576:
                    break
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2
            for a in selected:
                dp_w = dp_without[a]
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0 or target > 24576:
                        continue
                    if (dp_w[k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Valid, add x
            selected.append(x)
            
            # Save old state for x's exclusion set
            old_dp_all = dp_all
            
            # Update dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = old_dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For x, the set of others is the old selected set.
            new_dp_without[x] = old_dp_all + [0] * max(0, (n + 1) - len(old_dp_all))
            
            dp_without = new_dp_without
            n += 1
            
        return selected

    def validate_set(s):
        """Full bitset verification that no element is mean of subset of >=2 others."""
        n = len(s)
        if n < 3:
            return True
        
        # Build dp for full set
        dp_all = [1]
        for x in s:
            new_dp = [0] * (len(dp_all) + 1)
            new_dp[0] = 1
            for k in range(1, len(dp_all) + 1):
                val = dp_all[k-1] if k-1 < len(dp_all) else 0
                if k-1 < len(dp_all):
                    val |= (dp_all[k-1] << x)
                new_dp[k] = val
            dp_all = new_dp
        
        # For each element a, check if a is mean of any subset of size >= 2 from others
        for i, a in enumerate(s):
            others = s[:i] + s[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            # Build dp for others
            dp_others = [1]
            for x in others:
                new_dp = [0] * (len(dp_others) + 1)
                new_dp[0] = 1
                for k in range(1, len(dp_others) + 1):
                    val = dp_others[k-1] if k-1 < len(dp_others) else 0
                    if k-1 < len(dp_others):
                        val |= (dp_others[k-1] << x)
                    new_dp[k] = val
                dp_others = new_dp
            
            # Check all subset sizes k >= 2
            for k in range(2, m + 1):
                target = k * a
                if target > 24576:
                    break
                if (dp_others[k] >> target) & 1:
                    return False
        
        return True

    def local_search_full(initial_set, max_iters=500):
        """
        Local search on the full constraint.
        Tries to increase the size of the set by swapping elements.
        """
        current_set = list(initial_set)
        current_set.sort()
        current_len = len(current_set)
        all_candidates = set(range(1, 513))
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            if time.time() - start_time > deadline:
                break
            improved = False
            iters += 1
            
            # Try to add an element
            # To make this efficient, we can shuffle candidates or pick randomly
            candidates_to_try = list(all_candidates - set(current_set))
            random.shuffle(candidates_to_try)
            
            for new_x in candidates_to_try:
                if time.time() - start_time > deadline:
                    break
                
                # Check if we can just add new_x
                # This would require checking if new_x is mean of subset of current_set
                # AND if any a in current_set is mean of subset of (current_set + new_x) \ {a}
                
                # Fast check 1: Is new_x mean of subset of current_set?
                # We can use a quick DP for current_set
                # But building DP for every candidate is slow.
                # Let's just try the swap strategy directly.
                
                # Strategy: Try to swap one element out to make room for new_x
                # Or simply try to add new_x if current_len < 48.
                
                if current_len < 48:
                    # Try to add new_x
                    test_set = current_set + [new_x]
                    if validate_set(test_set):
                        current_set = test_set
                        current_len += 1
                        improved = True
                        break
                
                # If we couldn't add, try to swap
                # Pick a random element from current_set to remove
                # This is O(N) swaps to try per candidate, which is expensive.
                # Let's limit this.
                
        return current_set

    # Strategy 1: Multiple random restarts of the greedy approach
    orders = []
    
    # Spread
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(lo)
        if lo != hi:
            spread.append(hi)
        lo += 1
        hi -= 1
    orders.append(spread)
    
    # Random shuffles
    for i in range(50): # Increase restarts
        random.seed(i)
        cand = list(range(1, 513))
        random.shuffle(cand)
        orders.append(cand)
        
    # Ascending/Descending
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    for order in orders:
        if time.time() - start_time > deadline:
            break
        res = try_greedy_v2(order)
        
        # Apply local search to the result
        if len(res) > 0:
            res = local_search_full(res, max_iters=100)
            
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
                
    return best_set

# EVOLVE-BLOCK-END
