"""Combined TSP heuristic: 2-step lookahead + destination awareness + time-budgeted exact DP tail."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 140.0  # seconds (10s safety margin before 150s kill)

def _held_karp_next(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Finds the optimal next node for the remaining set using Held-Karp DP.
    Returns the first node in the optimal sequence.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    
    # Map unvisited nodes to local indices 0..n-1
    full_mask = (1 << n) - 1
    
    INF = float('inf')
    # dp[mask][i] = min cost to start at current_node, visit nodes in mask, end at unv[i]
    dp = np.full((1 << n, n), INF, dtype=float)
    parent = np.full((1 << n, n), -1, dtype=int)
    
    # Base cases: start at current_node, go to unv[i]
    for i in range(n):
        dp[1 << i, i] = distance_matrix[current_node, unv[i]]
        
    # Fill DP table
    # Iterate over all masks
    for mask in range(1 << n):
        # If mask is 0, skip
        if mask == 0:
            continue
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
        # Fallback
        return int(unv[0])
        
    # Backtrack to find the first node
    current_mask = full_mask
    current_i = best_last_i
    first_i = -1
    
    # We need to trace back to the mask with only 1 bit set.
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
    """
    Heuristic for larger n: 2-step lookahead blended with destination awareness.
    """
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])
    if n == 2:
        u1 = int(unv[0])
        u2 = int(unv[1])
        cost_u1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        cost_u2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        if cost_u1 <= cost_u2:
            return u1
        else:
            return u2

    dm = distance_matrix
    d_cur = dm[current_node, unv]

    sub = dm[np.ix_(unv, unv)]
    np.fill_diagonal(sub, np.inf)
    d_onward = sub.min(axis=1)

    two_step = d_cur + d_onward
    d_dest = dm[unv, destination_node]
    
    # Normalize components to [0, 1]
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

    # Blend: prioritize 2-step lookahead, but also consider distance to destination
    # and immediate distance. The 2-step term is the strongest signal.
    score = (0.60 * two_step_norm + 0.25 * d_cur_norm + 0.15 * d_dest_norm)
    idx = int(np.argmin(score))
    return int(unv[idx])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback to nearest neighbor if we're running out of time
        return int(unv[np.argmin(distance_matrix[current_node, unv])])
    
    # Structural Change: Use Exact DP for small remaining sets if time permits
    # DP is O(n^2 * 2^n). For n=15, that's ~15^2 * 32768 ~ 7.3M ops, which is fast.
    # For n=16, ~16^2 * 65536 ~ 16.7M ops.
    # We allow DP for n <= 15 to be safe on speed, but check time first.
    if n <= 15:
        # Estimate time for DP: roughly proportional to n^2 * 2^n
        # If we have plenty of time left, use DP.
        # 150s total, 8 instances, 50 calls each. ~400 calls total.
        # Average time per call should be < 0.3s to be safe.
        # DP for n=15 is usually < 0.1s in Python with numpy? Actually pure python loops might be slower.
        # Let's be conservative and only use DP if we have > 5s left and n <= 13.
        # Or just use it for n <= 12 to be very safe.
        if elapsed < _DEADLINE - 5.0 and n <= 13:
            return _held_karp_next(current_node, destination_node, unv, distance_matrix)
    
    return _select_heuristic(current_node, destination_node, unv, distance_matrix)
# EVOLVE-BLOCK-END
