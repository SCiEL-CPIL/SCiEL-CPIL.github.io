import numpy as np
import random
import time

def construct():
    start_time = time.time()
    max_time = 85.0  # Safety margin
    n = 512
    max_sum = n * 60  # Max possible sum for ~60 elements
    
    best_set = []
    best_size = 0
    
    # DP table for the current set
    # dp[k] is a numpy bool array of size max_sum + 1
    # dp[k][s] == True if there is a subset of size k with sum s
    
    def is_valid_addition(x, S, dp):
        # Check if x is the mean of a subset of S
        # Subset T of S, |T| >= 2, sum(T) = |T| * x
        for k in range(2, len(S) + 1):
            target_sum = k * x
            if target_sum > max_sum:
                break
            if dp[k][target_sum]:
                return False
        
        # Check if adding x makes any y in S the mean of a subset including x
        # New subset T' = T U {x}, T subset of S, |T| >= 1
        # sum(T) + x = (|T| + 1) * y
        # sum(T) = (|T| + 1) * y - x
        # Let k = |T|, so k >= 1.
        for y in S:
            for k in range(1, len(S)):
                target_sum = (k + 1) * y - x
                if target_sum < 0 or target_sum > max_sum:
                    continue
                if dp[k][target_sum]:
                    return False
        return True

    def update_dp(dp, x):
        # Update dp to include x
        # dp_new[k+1] = dp_old[k+1] OR (dp_old[k] << x)
        # We can do this in place if we iterate k downwards
        # But dp is a list of arrays. We need to be careful.
        # Actually, we can just create a new list or update in reverse.
        # Since we are in a greedy loop, we can just maintain the current dp.
        # But we need to backtrack if we fail? No, greedy doesn't backtrack.
        # So we can update in place if we iterate k from max down to 0.
        # However, dp[k] is used to compute dp[k+1].
        # If we update dp[k+1] using dp[k], and then update dp[k] using dp[k-1], it's fine.
        # But we need the OLD dp[k] to compute NEW dp[k+1].
        # So we should iterate k from max_size down to 0.
        # dp[k+1] |= (dp[k] << x)
        # But dp[k] on the RHS is the OLD dp[k] if we haven't updated it yet.
        # If we iterate k from max down to 0, then when we update dp[k+1], dp[k] is still old?
        # No, dp[k] is updated when we process k-1.
        # So if we go from high k to low k:
        # k = max: dp[max+1] |= dp[max] << x. dp[max] is old. Correct.
        # k = max-1: dp[max] |= dp[max-1] << x. dp[max-1] is old. Correct.
        # This works.
        
        # However, we need to know the current max size.
        # Let's just pass the size.
        pass # Placeholder, logic implemented in main loop

    # We will run multiple random restarts
    # To save time, we can try to find a good set quickly.
    
    for _ in range(1000):
        if time.time() - start_time > max_time:
            break
            
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = []
        # Initialize DP
        # Max size is at most 60
        max_k = 60
        dp = [np.zeros(max_sum + 1, dtype=bool) for _ in range(max_k + 1)]
        dp[0][0] = True
        
        current_max_k = 0
        
        for x in candidates:
            if time.time() - start_time > max_time:
                break
                
            if is_valid_addition(x, S, dp):
                # Update DP
                # We need to update dp for the new element x
                # Iterate k from current_max_k down to 0
                for k in range(current_max_k, -1, -1):
                    # dp[k+1] |= dp[k] << x
                    # Shift dp[k] left by x
                    # We can use np.roll or slicing
                    # dp[k] is bool array.
                    # shifted = np.zeros_like(dp[k])
                    # shifted[x:] = dp[k][:-x]
                    # dp[k+1] |= shifted
                    
                    # Optimization: only shift if x is within bounds
                    if x <= max_sum:
                        shifted = np.zeros(max_sum + 1, dtype=bool)
                        shifted[x:] = dp[k][:-x]
                        dp[k+1] |= shifted
                
                S.append(x)
                current_max_k += 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
