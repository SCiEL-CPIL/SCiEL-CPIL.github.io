import numpy as np
import random
import time

def construct():
    start_time = time.time()
    max_time = 85.0  # Safety margin
    n = 512
    max_size = 48
    max_sum = max_size * n + 10
    
    best_set = []
    best_size = 0
    
    # We will run multiple random restarts
    # To save time, we can try to find a good set quickly.
    
    # Pre-allocate DP arrays to avoid repeated allocation?
    # No, we need to reset them.
    # Creating 49 arrays of size 24676 is fast.
    
    num_restarts = 2000
    
    for _ in range(num_restarts):
        if time.time() - start_time > max_time:
            break
        if best_size == max_size:
            break
            
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = []
        # Initialize DP
        # dp[k] is a numpy bool array of size max_sum + 1
        # dp[k][s] == True if there is a subset of size k with sum s
        dp = [np.zeros(max_sum + 1, dtype=bool) for _ in range(max_size + 1)]
        dp[0][0] = True
        
        current_max_k = 0
        
        for x in candidates:
            if time.time() - start_time > max_time:
                break
            if current_max_k == max_size:
                break
                
            # Check if x is the mean of a subset of S
            # Subset T of S, |T| >= 2, sum(T) = |T| * x
            valid = True
            # k ranges from 2 to current_max_k
            for k in range(2, current_max_k + 1):
                target_sum = k * x
                if target_sum > max_sum:
                    break
                if dp[k][target_sum]:
                    valid = False
                    break
            
            if valid:
                # Update DP in place
                # Iterate k from current_max_k down to 0
                # dp[k+1] |= shift(dp[k], x)
                # dp[k+1][x:] |= dp[k][:-x]
                for k in range(current_max_k, -1, -1):
                    # x is always <= 512, max_sum is ~24600
                    # So x is well within bounds
                    dp[k+1][x:] |= dp[k][:-x]
                
                S.append(x)
                current_max_k += 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
