import numpy as np
import time

# Global cache for DP results? 
# The state is (current_node, destination_node, tuple(unvisited_nodes)).
# Since the order of unvisited nodes in the array might vary, we should sort them or use the tuple as key.
# However, caching might use too much memory if the search space is large.
# But for N=50, we only call this with N<=15 for the DP part.
# The number of unique (current, dest, unvisited_set) states with size <= 15 is manageable?
# Actually, the construction path is unique per instance. We don't branch. 
# We just call select_next_node sequentially.
# So caching is not strictly necessary for speedup across calls, as each call has a different unvisited set (shrinking).
# We just need to make the DP fast.

def _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Finds the optimal next node for the last k nodes using Held-Karp DP.
    Returns the first node in the optimal sequence.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Map unvisited nodes to local indices 0..n-1
    # We need to access distances between unvisited nodes and from current/dest to unvisited.
    
    # dp[mask][i] = min cost to start at current_node, visit nodes in mask, end at unv[i]
    # mask is a bitmask of length n.
    full_mask = (1 << n) - 1
    
    # Initialize DP table
    # Using a list of lists or a 2D numpy array?
    # n is small (<=15), so 2^n * n is small.
    # Python lists are slower than NumPy, but the logic is complex.
    # Let's use a 2D array of floats.
    INF = float('inf')
    dp = np.full((1 << n, n), INF, dtype=float)
    parent = np.full((1 << n, n), -1, dtype=int)
    
    # Base cases: start at current_node, go to unv[i]
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unv[i]]
        
    # Fill DP table
    # Iterate over all masks
    for mask in range(1 << n):
        # If mask is 0, skip (handled by base cases implicitly? No, base cases are mask with 1 bit)
        # We only transition from masks with >= 1 bit to masks with > 1 bit.
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            current_cost = dp[mask, i]
            if current_cost == INF:
                continue
            
            # Try extending to j not in mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                
                new_mask = mask | (1 << j)
                # Cost to go from unv[i] to unv[j]
                step_cost = distance_matrix[unv[i], unv[j]]
                new_cost = current_cost + step_cost
                
                if new_cost < dp[new_mask, j]:
                    dp[new_mask, j] = new_cost
                    parent[new_mask, j] = i

    # Find the best ending node i in full_mask such that dp[full_mask][i] + dist(unv[i], dest) is min
    best_cost = INF
    best_last_i = -1
    for i in range(n):
        total_cost = dp[full_mask, i] + distance_matrix[unv[i], destination_node]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_i = i
            
    if best_last_i == -1:
        # Fallback, shouldn't happen
        return int(unv[0])
        
    # Backtrack to find the first node
    current_mask = full_mask
    current_i = best_last_i
    first_i = -1
    
    # We need to trace back to the mask with only 1 bit set.
    # The parent array stores the previous node index.
    # However, the mask also changes. 
    # parent[new_mask][j] = i. This tells us the previous node was i.
    # But we also need to know the previous mask.
    # prev_mask = new_mask ^ (1 << j).
    
    path = []
    while current_mask != 0:
        path.append(current_i)
        if current_mask & (current_mask - 1) == 0:
            # Only one bit set, this is the first node
            first_i = current_i
            break
        
        prev_i = parent[current_mask, current_i]
        if prev_i == -1:
            # Error in backtracking
            break
        current_mask = current_mask ^ (1 << current_i)
        current_i = prev_i
        
    if first_i == -1:
        # Fallback
        first_i = 0
        
    return int(unv[first_i])


def _select_heuristic(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Original heuristic for larger n."""
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])

    dm = distance_matrix
    d_cur = dm[current_node, unv]

    sub = dm[np.ix_(unv, unv)]
    np.fill_diagonal(sub, np.inf)
    d_onward = sub.min(axis=1)

    two_step = d_cur + d_onward
    d_dest = dm[unv, destination_node]
    
    d_dest_range = d_dest.max() - d_dest.min()
    if d_dest_range > 1e-12:
        d_dest_norm = (d_dest - d_dest.min()) / d_dest_range
    else:
        d_dest_norm = np.zeros_like(d_dest)

    two_step_span = two_step.max() - two_step.min()
    if two_step_span > 1e-12:
        two_step_norm = (two_step - two_step.min()) / two_step_span
    else:
        two_step_norm = np.zeros_like(two_step)

    d_cur_span = d_cur.max() - d_cur.min()
    if d_cur_span > 1e-12:
        d_cur_norm = (d_cur - d_cur.min()) / d_cur_span
    else:
        d_cur_norm = np.zeros_like(d_cur)

    score = (0.70 * two_step_norm + 0.20 * d_cur_norm + 0.10 * d_dest_norm)
    idx = int(np.argmin(score))
    return int(unv[idx])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    # Structural Change: Use Exact DP for small remaining sets
    if n <= 15:
        return _held_karp_next(current_node, destination_node, unv, distance_matrix)
    else:
        return _select_heuristic(current_node, destination_node, unv, distance_matrix)
