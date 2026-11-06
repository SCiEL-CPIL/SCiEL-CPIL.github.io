import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    # Filter feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    # We use a Dynamic Programming approach on the remaining feasible items
    # to find an item that belongs to an optimal solution for the current state.
    
    # Scale to integers for DP
    scale = 100
    cap_int = int(remaining_capacity * scale + 0.5)
    
    # Prepare lists for DP
    # We only consider feasible items to reduce N
    weights_int = [int(remaining_weights[i] * scale + 0.5) for i in feasible_indices]
    values = [remaining_values[i] for i in feasible_indices]
    
    m = len(weights_int)
    if m == 0:
        return -1
        
    # DP Table: dp[i][c] = max value using first i items with capacity c
    # choice[i][c] = True if item i-1 is included in the optimal solution for dp[i][c]
    # To save memory and time, we can use 1D DP for value, but we need to backtrack.
    # Storing full 2D table is safe for m=100, cap=1250.
    
    # Optimization: If m is small, 2D is fine.
    dp = [[0] * (cap_int + 1) for _ in range(m + 1)]
    choice = [[False] * (cap_int + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        wi = weights_int[i-1]
        vi = values[i-1]
        for c in range(cap_int + 1):
            if wi <= c:
                val_take = dp[i-1][c-wi] + vi
                val_not = dp[i-1][c]
                if val_take > val_not:
                    dp[i][c] = val_take
                    choice[i][c] = True
                else:
                    dp[i][c] = val_not
            else:
                dp[i][c] = dp[i-1][c]
                
    # Backtrack to find one item in the optimal set
    c = cap_int
    for i in range(m, 0, -1):
        if choice[i][c]:
            # Item i-1 (in the reduced list) is in the solution
            # Map back to original index
            return feasible_indices[i-1]
        # If not chosen, capacity doesn't change
        # c remains c
        
    # If no item chosen (should not happen if cap > 0 and feasible items exist)
    return -1
