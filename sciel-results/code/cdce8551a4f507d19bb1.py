import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    if remaining_capacity <= 0:
        return -1
        
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Scale weights and capacity to integers
    # Using scale 100 to handle 2 decimal places
    scale = 100
    C = int(round(remaining_capacity * scale))
    if C <= 0:
        return -1
        
    # Convert weights to int
    # If a weight is 0, it's always feasible.
    # We can filter out items with weight > C immediately? 
    # DP handles it, but we can optimize by only considering feasible items?
    # No, keep indices aligned.
    
    int_weights = [int(round(w * scale)) for w in remaining_weights]
    
    # DP Table
    # dp[i][j] max value using first i items with capacity j
    # To save memory/time, we can use 1D array, but we need traceback.
    # 2D array is N+1 x C+1.
    # N <= 100, C <= 1250. Size ~ 126k.
    
    dp = [[0] * (C + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        w = int_weights[i-1]
        v = remaining_values[i-1]
        prev_row = dp[i-1]
        curr_row = dp[i]
        
        # Optimization: if w > C, item can't be picked.
        if w > C:
            curr_row[:] = prev_row[:]
        else:
            # Copy previous row
            curr_row[:] = prev_row[:]
            # Update
            for j in range(w, C + 1):
                val = prev_row[j-w] + v
                if val > curr_row[j]:
                    curr_row[j] = val
                    
    # Traceback
    j = C
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i-1][j]:
            # Item i-1 is in the optimal solution
            return i - 1
        # Else, item i-1 is not in this particular optimal solution path
        # (or it could be, but we found a path without it). 
        # We continue to i-1.
        # Note: j remains the same because we didn't subtract weight.
        
    return -1
