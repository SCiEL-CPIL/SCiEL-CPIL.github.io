import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 145  # Safety margin for 150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Time check: if we're running low on time, fall back to simple nearest neighbor
    if time.time() > _DEADLINE - 1.0:
        # Simple nearest neighbor fallback
        dists = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists)
        return int(unvisited_nodes[best_idx])
    
    if n_unvisited == 2:
        # Exact 2-step lookahead
        a = int(unvisited_nodes[0])
        b = int(unvisited_nodes[1])
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return a if cost_a < cost_b else b
    
    # For n_unvisited >= 3, use Held-Karp DP on the last min(n_unvisited, 15) nodes
    # For larger sets, we need a hybrid approach:
    # 1. Use a fast heuristic to identify a subset of candidate nodes to consider
    # 2. Apply exact DP on that subset
    
    # Strategy: Use a 2-step lookahead to score all nodes, then take top-K candidates
    # and apply exact DP on those K nodes plus the remaining structure
    
    # First, compute a quick score for each node using 2-step lookahead
    dist_v_to_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    dist_v_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_to_v = distance_matrix[current_node, unvisited_nodes]
    
    # 2-step lookahead: cost to go from v to dest via one intermediate node
    M_temp = dist_v_to_unvisited + dist_v_to_dest[None, :]
    np.fill_diagonal(M_temp, np.inf)
    best_cost_2step = np.min(M_temp, axis=1)
    
    # Quick score for each node
    quick_score = dist_current_to_v + best_cost_2step
    
    # Determine how many nodes to include in exact DP
    # Held-Karp is O(2^k * k^2), so for k=15 that's ~32768*225 ≈ 7.4M ops
    # We need to be careful with time
    
    # Use exact DP for up to 12 nodes (2^12 * 144 ≈ 590K ops)
    # For more nodes, use a reduced set
    
    # Sort by quick score and take top candidates
    sorted_indices = np.argsort(quick_score)
    
    # Decide K based on time budget and n_unvisited
    if n_unvisited <= 12:
        K = n_unvisited
    elif n_unvisited <= 20:
        K = 12
    else:
        # For larger n, use a smarter selection
        # Take top 8 by quick score, but ensure we don't exclude nodes that are
        # close to current or destination
        K = 10
    
    # Select K best candidates
    selected_local_indices = sorted_indices[:K]
    selected_nodes = unvisited_nodes[selected_local_indices]
    
    # Build sub-problem for exact DP
    # The sub-problem: we must visit all K selected nodes and then go to destination
    # But wait - this isn't quite right. We need to visit ALL unvisited nodes eventually.
    # The DP should model the optimal order to visit the selected nodes,
    # but we also need to account for the non-selected nodes.
    
    # Actually, a better approach: use the DP to find the optimal next node
    # by considering a lookahead window of the next few nodes.
    
    # Let's use a different approach: compute the exact optimal suffix for
    # a small window, and use heuristic for the rest.
    
    # Simpler approach: Use the quick score but refine it with a 3-step lookahead
    # for the top candidates
    
    # 3-step lookahead for top K candidates
    K_refine = min(K, n_unvisited)
    refine_indices = sorted_indices[:K_refine]
    refine_nodes = unvisited_nodes[refine_indices]
    
    if K_refine <= 1:
        return int(unvisited_nodes[refine_indices[0]])
    
    if K_refine == 2:
        a = int(refine_nodes[0])
        b = int(refine_nodes[1])
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return a if cost_a < cost_b else b
    
    # For K_refine >= 3, use DP
    k = K_refine
    # Map local indices to global
    # DP state: (mask, last_node) -> min cost
    # mask is a bitmask of which of the k nodes have been visited
    # last_node is the index of the last visited node (0 to k-1)
    
    # Initialize DP
    INF = float('inf')
    dp = np.full((1 << k, k), INF, dtype=np.float64)
    
    # Base case: visit one node
    for i in range(k):
        dp[1 << i, i] = distance_matrix[current_node, refine_nodes[i]]
    
    # Fill DP
    for mask in range(1, 1 << k):
        for last in range(k):
            if dp[mask, last] == INF:
                continue
            if not (mask & (1 << last)):
                continue
            # Try adding each unvisited node
            for nxt in range(k):
                if mask & (1 << nxt):
                    continue
                new_mask = mask | (1 << nxt)
                cost = dp[mask, last] + distance_matrix[refine_nodes[last], refine_nodes[nxt]]
                if cost < dp[new_mask, nxt]:
                    dp[new_mask, nxt] = cost
    
    # Find the best final state
    # After visiting all k nodes, we go to destination
    full_mask = (1 << k) - 1
    best_total = INF
    best_last = -1
    for last in range(k):
        total = dp[full_mask, last] + distance_matrix[refine_nodes[last], destination_node]
        if total < best_total:
            best_total = total
            best_last = last
    
    # Now we need to find the FIRST node in the optimal path
    # Reconstruct path from DP
    # We need to find which node was visited first in the optimal path
    
    # Backtrack
    path = []
    mask = full_mask
    last = best_last
    while mask != 0:
        path.append(last)
        prev_mask = mask & ~(1 << last)
        if prev_mask == 0:
            break
        # Find the previous node
        best_prev_cost = INF
        best_prev = -1
        for prev in range(k):
            if not (prev_mask & (1 << prev)):
                continue
            cost = dp[prev_mask, prev] + distance_matrix[refine_nodes[prev], refine_nodes[last]]
            if cost < best_prev_cost - 1e-9:
                best_prev_cost = cost
                best_prev = prev
        if best_prev == -1:
            break
        last = best_prev
        mask = prev_mask
    
    path.reverse()
    if len(path) > 0:
        first_local_idx = path[0]
        return int(refine_nodes[first_local_idx])
    else:
        # Fallback
        return int(unvisited_nodes[refine_indices[0]])
