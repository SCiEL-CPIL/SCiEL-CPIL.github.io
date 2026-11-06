"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE = 115.0


def _elapsed():
    return time.time() - _start_time


def _time_exceeded():
    return _elapsed() > _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Deadline guard: if we're running low on time, fall back to nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_cities = distance_matrix.shape[0]

    # For small sets (<= 15), use Held-Karp DP for exact optimal suffix
    if n_unvisited <= 15:
        return _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix)

    # For larger sets, use a lookahead heuristic with stranding and destination awareness
    return _lookahead_select(current_node, destination_node, unvisited_nodes, distance_matrix)


def _held_karp_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Use Held-Karp DP to find the optimal next node for small sets (n <= 15).
    
    dp[mask][j] = min cost to start at current_node, visit all nodes in mask, and end at node_list[j].
    Base: dp[1<<j][j] = dist(current_node, node_list[j])
    Recurrence: dp[mask][j] = min over i in mask, i != j: dp[mask ^ (1<<j)][i] + dist(node_list[i], node_list[j])
    
    Total cost for ending at j = dp[all_mask][j] + dist(node_list[j], destination_node)
    We want to minimize this over j, and return the first node in the optimal path.
    
    To find the first node: for each candidate first node j, the total cost is:
    dist(current, j) + min_cost_to_visit_remaining_from_j_and_end_at_destination
    
    We compute the full DP, then backtrack to find which j gives the minimum total.
    """
    n = len(unvisited_nodes)
    node_list = unvisited_nodes.astype(int)
    
    # Precompute distances between unvisited nodes
    # dist_mat[i][j] = distance between node_list[i] and node_list[j]
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i][j] = distance_matrix[node_list[i], node_list[j]]
    
    # dp[mask][j] = min cost to start at current_node, visit all nodes in mask, and end at node_list[j]
    # mask is a bitmask of which nodes have been visited
    # j must be in mask
    INF = float('inf')
    dp = np.full((1 << n, n), INF)
    
    # Base case: visit only node j
    for j in range(n):
        dp[1 << j][j] = distance_matrix[current_node, node_list[j]]
    
    # Fill in the DP table
    for mask in range(1, 1 << n):
        for j in range(n):
            if not (mask & (1 << j)):
                continue
            if mask == (1 << j):
                continue
            # Try all possible previous nodes i
            best = INF
            for i in range(n):
                if i == j:
                    continue
                if not (mask & (1 << i)):
                    continue
                prev_mask = mask ^ (1 << j)
                if dp[prev_mask][i] < best:
                    # Check if i is in prev_mask
                    if prev_mask & (1 << i):
                        val = dp[prev_mask][i] + dist_mat[i][j]
                        if val < best:
                            best = val
            dp[mask][j] = best
    
    # Find the best ending node j
    full_mask = (1 << n) - 1
    best_total = INF
    best_end_j = -1
    for j in range(n):
        total = dp[full_mask][j] + distance_matrix[node_list[j], destination_node]
        if total < best_total:
            best_total = total
            best_end_j = j
    
    # Backtrack to find the first node
    # We need to find which node was visited first in the optimal path
    mask = full_mask
    j = best_end_j
    path = [j]
    while mask != (1 << j):
        prev_mask = mask ^ (1 << j)
        best_prev = -1
        best_val = INF
        for i in range(n):
            if i == j:
                continue
            if not (prev_mask & (1 << i)):
                continue
            val = dp[prev_mask][i] + dist_mat[i][j]
            if val < best_val:
                best_val = val
                best_prev = i
        path.append(best_prev)
        j = best_prev
        mask = prev_mask
    
    path.reverse()
    first_node = path[0]
    return int(node_list[first_node])


def _lookahead_select(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    For larger sets, use a lookahead heuristic:
    - Compute distance from current to each candidate
    - Compute distance from each candidate to destination
    - Compute stranding penalty (max distance from current to other unvisited nodes)
    - Combine these with adaptive weights
    """
    n_unvisited = len(unvisited_nodes)
    n_cities = distance_matrix.shape[0]
    
    # Distances from current node to all unvisited candidates
    d_current_cand = distance_matrix[current_node, unvisited_nodes]
    
    # Distances from each candidate to the destination
    d_cand_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Fraction of tour remaining (1.0 at start, ~0 near end)
    frac_remaining = n_unvisited / n_cities
    
    # Stranding penalty: for each candidate, max distance from current to OTHER unvisited nodes
    d_sorted = np.sort(d_current_cand)
    global_max = d_sorted[-1]
    second_max = d_sorted[-2] if n_unvisited >= 2 else 0.0
    argmax_idx = np.argmax(d_current_cand)
    
    # For each candidate index i, max_excluding_i = global_max if i != argmax_idx else second_max
    max_excluding = np.where(np.arange(n_unvisited) == argmax_idx, second_max, global_max)
    
    # Weights: stranding penalty moderate; destination penalty grows as we near end
    w_strand = 0.5
    w_dest = 1.0 * (1.0 - frac_remaining)
    
    # Total score for each candidate
    scores = d_current_cand + w_strand * max_excluding + w_dest * d_cand_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
