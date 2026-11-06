"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _time_left():
    if _START_TIME is None:
        return True
    # Safety margin: return before 145s of 150s limit
    return (time.time() - _START_TIME) < 145.0


def _simulate_greedy_value(capacity, weights, values):
    """Simulate greedy fill by value density. Returns total value."""
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    # Filter to feasible items (weight <= capacity)
    feasible = weights <= capacity
    if not np.any(feasible):
        return 0.0
    w = weights[feasible]
    v = values[feasible]
    # Sort by density descending
    order = np.argsort(-v / w)
    w = w[order]
    v = v[order]
    total = 0.0
    remaining = capacity
    for wi, vi in zip(w, v):
        if wi <= remaining:
            remaining -= wi
            total += vi
    return total


def _dp_optimal_value(capacity, weights, values, max_capacity_int):
    """
    Compute optimal knapsack value using DP on scaled/integer weights.
    Returns (optimal_value, first_item_on_optimal_path).
    first_item is -1 if no items can be added.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0, -1
    
    # Scale weights to integers for DP
    # Use a scaling factor to keep DP table manageable
    # For capacity ~12.5 and 100 items, weights are likely small
    # Let's use integer weights by rounding up to ensure feasibility
    
    # First, check if we can use integer DP directly
    # If weights are already integers or close to it
    w_int = np.ceil(weights).astype(int)
    
    # Check if scaling is needed
    # If max weight is too large, we might need to scale down
    max_w = int(np.max(w_int)) if n > 0 else 0
    
    # For capacity ~12.5, if max_w is reasonable, we can do DP
    # DP table size: (n+1) x (capacity_int+1)
    cap_int = int(np.ceil(capacity))
    
    # Limit DP table size to avoid memory issues
    # If cap_int * n is too large, fall back to greedy
    if cap_int * n > 500000:
        return _simulate_greedy_value(capacity, weights, values), -1
    
    # Build DP table
    # dp[i][c] = max value using first i items with capacity c
    # We want to find the first item on the optimal path
    # dp[i][c] = max(dp[i-1][c], dp[i-1][c-w[i]] + v[i]) if w[i] <= c
    
    # To find the first item on the optimal path, we can do backward DP
    # or forward DP and track choices
    
    # Let's do standard DP and then backtrack to find the first item
    # Actually, we want the first item that appears in the optimal solution
    # among the remaining items
    
    # dp[c] for current row, dp_prev[c] for previous row
    # We'll do it in a way that lets us backtrack
    
    # Better approach: compute dp[i][c] for all i, c
    # Then backtrack from dp[n][cap_int] to find which items are included
    # The first item (lowest index in original order) that is included is our answer
    
    # But we need to be careful: the "first" item on the optimal path
    # should be the one with the smallest original index that is part of the optimal solution
    
    # Let's use a 2D DP table if size permits
    if cap_int * n > 500000:
        return _simulate_greedy_value(capacity, weights, values), -1
    
    # dp[i][c] = max value using items 0..i-1 with capacity c
    dp = np.zeros((n + 1, cap_int + 1), dtype=float)
    
    for i in range(1, n + 1):
        wi = w_int[i - 1]
        vi = values[i - 1]
        for c in range(cap_int + 1):
            dp[i][c] = dp[i - 1][c]
            if wi <= c:
                candidate = dp[i - 1][c - wi] + vi
                if candidate > dp[i][c]:
                    dp[i][c] = candidate
    
    optimal_val = dp[n][cap_int]
    
    # Backtrack to find which items are in the optimal solution
    included = [False] * n
    c = cap_int
    for i in range(n, 0, -1):
        wi = w_int[i - 1]
        if wi <= c and dp[i][c] == dp[i - 1][c - wi] + values[i - 1]:
            included[i - 1] = True
            c -= wi
    
    # Find the first included item (smallest index)
    first_item = -1
    for i in range(n):
        if included[i]:
            # Check if this item is actually feasible in the current remaining state
            # (it should be, since we built the DP on remaining items)
            first_item = i
            break
    
    return optimal_val, first_item


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    _init_time()
    
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
    
    # Check time
    if not _time_left():
        return -1
    
    # Filter to feasible items
    feasible_mask = remaining_weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    # Get feasible indices
    feasible_indices = np.where(feasible_mask)[0]
    n_feasible = len(feasible_indices)
    
    if n_feasible == 0:
        return -1
    
    w_feasible = remaining_weights[feasible_mask]
    v_feasible = remaining_values[feasible_mask]
    
    # Try DP approach if feasible
    cap_int = int(np.ceil(remaining_capacity))
    
    # DP is feasible if the table isn't too large
    if cap_int * n_feasible <= 500000:
        try:
            opt_val, first_local = _dp_optimal_value(remaining_capacity, w_feasible, v_feasible, cap_int)
            if first_local != -1:
                return int(feasible_indices[first_local])
        except Exception:
            pass
    
    # Fallback: greedy by value density
    densities = v_feasible / w_feasible
    best_local = int(np.argmax(densities))
    return int(feasible_indices[best_local])
# EVOLVE-BLOCK-END
