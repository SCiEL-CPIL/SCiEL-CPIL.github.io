import numpy as np
import random
import time

def construct():
    start_time = time.time()
    max_time = 85.0  # Safety margin
    n = 512
    max_size = 48
    max_sum = max_size * n + 100  # 24676

    # DP arrays: dp[k][s] is True if subset of size k has sum s
    # We will maintain a DP for the current set S
    dp = [np.zeros(max_sum + 1, dtype=bool) for _ in range(max_size + 1)]
    
    def build_dp(S):
        """Build DP table from scratch for set S."""
        for k in range(max_size + 1):
            dp[k][:] = False
        dp[0][0] = True
        current_max_k = 0
        for x in S:
            # Update DP in reverse k
            for k in range(current_max_k, -1, -1):
                if x <= max_sum:
                    dp[k+1][x:] |= dp[k][:-x]
            current_max_k += 1
        return current_max_k

    def is_valid_add(x, S, dp_table, current_max_k):
        """
        Check if x can be added to S, given dp_table of S.
        Returns True if valid.
        """
        # 1. x is not mean of any subset of S (size >= 2)
        for k in range(2, current_max_k + 1):
            target = k * x
            if target > max_sum:
                break
            if dp_table[k][target]:
                return False
        
        # 2. No existing y in S is mean of subset of S U {x} including x
        # mean(T U {x}) = y => sum(T) + x = (k+1)y => sum(T) = (k+1)y - x
        # T is subset of S, |T|=k >= 1
        for y in S:
            for k in range(1, current_max_k + 1):
                target = (k + 1) * y - x
                if target < 0 or target > max_sum:
                    continue
                if dp_table[k][target]:
                    return False
        return True

    def update_dp_add(x, current_max_k):
        """Update dp in place to include x."""
        for k in range(current_max_k, -1, -1):
            if x <= max_sum:
                dp[k+1][x:] |= dp[k][:-x]

    # 1. Initial Greedy Construction
    S = []
    current_max_k = 0
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    
    # Build initial DP (empty set)
    for k in range(max_size + 1):
        dp[k][:] = False
    dp[0][0] = True

    for x in candidates:
        if time.time() - start_time > max_time:
            break
        if is_valid_add(x, S, dp, current_max_k):
            update_dp_add(x, current_max_k)
            S.append(x)
            current_max_k += 1
            if current_max_k == max_size:
                break

    best_size = len(S)
    best_set = S[:]

    # 2. Local Search (Hill Climbing)
    # We try to improve the set by swapping elements or adding if possible.
    # Since we can't easily remove from DP, we rebuild DP for S\{y} when considering swaps.
    
    all_candidates = set(range(1, n + 1))
    
    while time.time() - start_time < max_time:
        # Try to add any remaining element
        added = False
        remaining = [x for x in all_candidates if x not in S]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() - start_time > max_time:
                break
            if is_valid_add(x, S, dp, current_max_k):
                update_dp_add(x, current_max_k)
                S.append(x)
                current_max_k += 1
                added = True
                break
        
        if added:
            if current_max_k > best_size:
                best_size = current_max_k
                best_set = S[:]
            continue
        
        # If no element could be added, try a swap
        # Pick a random y to remove
        if not S:
            break
            
        y = random.choice(S)
        
        # Build DP for S without y
        S_temp = [x for x in S if x != y]
        dp_temp = [np.zeros(max_sum + 1, dtype=bool) for _ in range(max_size + 1)]
        dp_temp[0][0] = True
        k_temp = 0
        for x in S_temp:
            for k in range(k_temp, -1, -1):
                if x <= max_sum:
                    dp_temp[k+1][x:] |= dp_temp[k][:-x]
            k_temp += 1
        
        # Try to add a random x from remaining
        found_swap = False
        # To save time, don't try ALL remaining, just a random sample or all if fast
        # Since k_temp is ~47, is_valid_add is fast.
        # Let's try a few random swaps.
        num_attempts = 20
        candidates_to_try = random.sample(remaining, min(len(remaining), num_attempts))
        
        for x in candidates_to_try:
            if time.time() - start_time > max_time:
                break
            if is_valid_add(x, S_temp, dp_temp, k_temp):
                # Swap y with x
                S.remove(y)
                S.append(x)
                # Update main DP to reflect S
                # We can just rebuild main dp from S_temp + x
                # Or update dp_temp to include x and copy
                # Copying dp_temp (48 arrays of 24k bools) is ~1.2MB, fast enough.
                # Let's just update dp_temp to include x and assign
                for k in range(k_temp, -1, -1):
                    if x <= max_sum:
                        dp_temp[k+1][x:] |= dp_temp[k][:-x]
                
                # Copy dp_temp to dp
                for k in range(max_size + 1):
                    dp[k][:] = dp_temp[k]
                
                current_max_k = k_temp + 1
                found_swap = True
                break
        
        if not found_swap:
            # Stuck? Maybe try a different y?
            # To avoid getting stuck in a local minimum, we can do a random restart if no progress for a while.
            # But for now, just continue the loop. It will pick a new y.
            pass

    return best_set
