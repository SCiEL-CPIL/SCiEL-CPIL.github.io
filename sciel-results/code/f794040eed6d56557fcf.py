"""Rewritten TSP constructive heuristic with lookahead and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard (120s limit, use 110s for safety)
_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0


def _time_exceeded():
    return time.time() > _DEADLINE


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city in a TSP tour construction.
    
    Uses a hybrid strategy:
    - For large unvisited sets: greedy nearest neighbor with 1-step lookahead
    - For small unvisited sets (<=15): exact Held-Karp DP for optimal suffix
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen, but return destination as fallback
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If time is nearly up, fall back to simple nearest neighbor
    if _time_exceeded():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small remaining sets, use exact DP (Held-Karp) to find optimal next step
    if n_unvisited <= 15:
        # Find the best next node such that the optimal completion from there is minimized
        # dp[mask][last] = min cost to visit all nodes in mask ending at last
        # We want to find which node to go to next (from current_node) that leads to best total
        
        m = n_unvisited
        # Map unvisited_nodes indices to 0..m-1
        node_map = {unvisited_nodes[i]: i for i in range(m)}
        
        # dp[mask][j] = min cost to visit exactly the set of nodes indicated by mask, ending at node j (0-indexed in unvisited)
        # We start by choosing a first node from current_node, then complete the rest
        
        # Actually, we need: for each candidate first node k, compute:
        #   dist(current, k) + optimal_tour_cost_from_k_visiting_remaining(m-1 nodes)
        # The optimal tour cost from k visiting all other m-1 nodes and returning to destination_node
        
        # Held-Karp: dp[mask][j] where mask is subset of {0..m-1} and j in mask
        # dp[mask][j] = min over i in mask\{j} of dp[mask\{j}][i] + dist(unvisited[i], unvisited[j])
        # Base: dp[{j}][j] = dist(current_node, unvisited[j])
        
        # Full mask is (1<<m) - 1
        full_mask = (1 << m) - 1
        
        # We'll compute dp for all masks
        # dp[mask][j] for j in mask
        # Use a 2D array: dp[1<<m][m], initialized to inf
        # This is 2^15 * 15 = 491520 entries, manageable
        
        INF = float('inf')
        dp = np.full((1 << m, m), INF, dtype=np.float64)
        
        # Base cases: single node in mask
        for j in range(m):
            dp[1 << j, j] = distance_matrix[current_node, unvisited_nodes[j]]
        
        # Fill in DP
        for mask in range(1, 1 << m):
            for j in range(m):
                if not (mask & (1 << j)):
                    continue
                if mask == (1 << j):
                    continue  # already set
                # Try all i in mask \ {j}
                remaining = mask & ~(1 << j)
                # Iterate over set bits in remaining
                r = remaining
                while r:
                    low_bit = r & (-r)
                    i = int(np.log2(low_bit))
                    if dp[mask ^ (1 << j), i] + distance_matrix[unvisited_nodes[i], unvisited_nodes[j]] < dp[mask, j]:
                        dp[mask, j] = dp[mask ^ (1 << j), i] + distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
                    r ^= low_bit
        
        # Now for each candidate first node k, the total cost is:
        # dp[full_mask][k] + dist(unvisited[k], destination_node)
        # Wait, that's not quite right. dp[full_mask][k] already includes the path from current through all nodes ending at k.
        # Then we add dist from k back to destination_node.
        
        best_cost = INF
        best_node_idx = 0
        for k in range(m):
            total = dp[full_mask, k] + distance_matrix[unvisited_nodes[k], destination_node]
            if total < best_cost:
                best_cost = total
                best_node_idx = k
        
        return int(unvisited_nodes[best_node_idx])
    
    # For larger sets, use greedy with 1-step lookahead
    # Score each candidate u: dist(current, u) + min_{v in unvisited\{u}} dist(u, v)
    # This is O(n^2) which is fine for n=50
    
    distances_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate, compute the minimum distance from that candidate to any other unvisited node
    # This is the "lookahead" term
    # dist_matrix among unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # min_dist_from_u_to_others[u] = min over v != u of sub_dist[u, v]
    # We can compute this by finding the second smallest in each row (since we exclude self)
    # Actually, just set diagonal to inf and take row min
    min_lookahead = np.full(n_unvisited, INF, dtype=np.float64)
    for i in range(n_unvisited):
        # Find min distance from unvisited[i] to any other unvisited node
        # Exclude i itself
        if n_unvisited > 1:
            min_lookahead[i] = np.min(np.delete(sub_dist[i], i))
    
    # Combined score: distance from current + lookahead
    scores = distances_from_current + min_lookahead
    
    # Pick the node with minimum score
    best_idx = int(np.argmin(scores))
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
