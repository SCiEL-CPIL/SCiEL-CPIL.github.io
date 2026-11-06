import numpy as np
import time

_START = time.monotonic()
_TOTAL_BUDGET = 110.0
_SCALE = 1000

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item to add to the knapsack using exact Dynamic Programming
    on the remaining subproblem.
    """
    elapsed = time.monotonic() - _START
    if elapsed > _TOTAL_BUDGET:
        return -1

    n = len(remaining_weights)
    if n == 0:
        return -1

    if remaining_capacity <= 0:
        return -1

    # Convert to numpy arrays for speed
    w_rem = np.asarray(remaining_weights, dtype=float)
    v_rem = np.asarray(remaining_values, dtype=float)

    # Scale capacity and weights to integers
    cap_int = int(round(remaining_capacity * _SCALE))
    if cap_int <= 0:
        return -1
        
    w_int = np.round(w_rem * _SCALE).astype(int)
    
    # Filter out items that are too heavy to fit in the remaining capacity
    feasible_mask = w_int <= cap_int
    if not np.any(feasible_mask):
        return -1

    # Get indices of feasible items in the original remaining arrays
    feasible_indices = np.where(feasible_mask)[0]
    w_f = w_int[feasible_indices]
    v_f = v_rem[feasible_indices]
    m = len(w_f)

    if m == 1:
        return int(feasible_indices[0])

    # DP Table: dp[i][c] = max value using items i..m-1 with capacity c
    # We allocate (m+1) x (cap_int+1)
    dp = np.zeros((m + 1, cap_int + 1), dtype=float)
    
    # Fill DP table backwards
    # dp[m][c] = 0 for all c
    for i in range(m - 1, -1, -1):
        wi = w_f[i]
        vi = v_f[i]
        
        # Start with the case where we don't take item i
        # dp[i][:] = dp[i+1][:]
        dp[i][:] = dp[i+1][:]
        
        # Update with the case where we take item i
        if wi <= cap_int:
            # dp[i][wi:] = max(dp[i][wi:], dp[i+1][:cap_int-wi] + vi)
            # Note: dp[i+1][:cap_int-wi] has length cap_int - wi + 1
            # dp[i][wi:] has length cap_int - wi + 1
            # So the shapes match.
            dp[i][wi:] = np.maximum(dp[i][wi:], dp[i+1][:cap_int - wi] + vi)

    # Backtrack to find the first item in the optimal solution
    c = cap_int
    for i in range(m):
        wi = w_f[i]
        # If dp[i][c] is strictly greater than dp[i+1][c], item i must be in the optimal solution
        if dp[i][c] > dp[i+1][c] + 1e-9:
            # Return the index in the original remaining_weights array
            return int(feasible_indices[i])
            
    # If no item was found with strict improvement, it means the optimal value is 0
    # or there is a tie. In a 0/1 knapsack with positive values, if optimal > 0,
    # there must be an item that improves the value.
    # If optimal is 0, we return -1.
    # However, due to floating point, we might have missed it.
    # Let's check if dp[0][cap_int] is significantly greater than 0.
    if dp[0][cap_int] > 1e-9:
        # Fallback: pick the item with highest value that fits
        # This should not happen if the logic above is correct, but safe.
        best_idx = -1
        best_val = -1
        for i in range(m):
            if w_f[i] <= cap_int and v_f[i] > best_val:
                best_val = v_f[i]
                best_idx = i
        if best_idx != -1:
            return int(feasible_indices[best_idx])
            
    return -1
