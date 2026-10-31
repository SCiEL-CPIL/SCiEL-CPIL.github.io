"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _solve_small_tsp(start, nodes, end, dist_matrix):
    """
    Solves the small TSP problem: find the optimal path from 'start' 
    visiting all 'nodes' and ending at 'end'.
    Returns the index (in the original node space) of the first node to visit.
    Uses DP with bitmasks.
    """
    n = len(nodes)
    if n == 0:
        return -1
    if n == 1:
        return nodes[0]
    
    # Map nodes to 0..n-1 indices for bitmask
    # dp[mask][i] = min cost to start at 'start', visit set 'mask' (subset of nodes), 
    # and end at nodes[i]
    
    # Optimization: We only need to find the best first step.
    # However, calculating the full DP gives us the optimal suffix cost for any first step.
    
    # dist_matrix is global. We need distances between:
    # start -> nodes[i]
    # nodes[i] -> nodes[j]
    # nodes[i] -> end
    
    # Precompute local distance matrix for the subproblem
    # 0 is start, 1..n are nodes, n+1 is end? 
    # Simpler: use the original indices and a dictionary or direct lookup.
    
    # Let's use a dictionary for memoization or a 2D array.
    # n is small (<= 10), so 2^n * n is fine.
    
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    # parent tracking to reconstruct path? We only need the first node.
    # But to compute dp, we need the costs.
    
    # Initialize: start -> nodes[i]
    for i in range(n):
        dp[1 << i][i] = dist_matrix[start, nodes[i]]
        
    # Fill DP table
    for mask in range(1, 1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == INF:
                continue
            # Try adding node j
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist_matrix[nodes[i], nodes[j]]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    
    # Find the best completion from any node i in the full set to 'end'
    full_mask = (1 << n) - 1
    best_cost = INF
    best_last_node_idx = -1
    for i in range(n):
        total_cost = dp[full_mask][i] + dist_matrix[nodes[i], end]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_node_idx = i
            
    # Now we have the optimal total cost. We need to find which node was the FIRST one visited.
    # We can backtrack from the optimal state (full_mask, best_last_node_idx)
    # to find the first node.
    
    # Alternatively, we can just evaluate the cost of starting with each node k.
    # Cost(k) = dist(start, nodes[k]) + min_cost_to_visit_rest_and_end(nodes[k])
    # This is exactly what the DP gives us if we look at the transitions.
    # Actually, the simplest way to get the first node is to check which initial choice leads to the optimal solution.
    
    # Let's backtrack.
    curr_mask = full_mask
    curr_node = best_last_node_idx
    path = [curr_node]
    
    while curr_mask != 0:
        # Find predecessor
        prev_mask = curr_mask
        # We need to find i such that dp[prev_mask ^ (1<<curr_node)][i] + dist(nodes[i], nodes[curr_node]) == dp[curr_mask][curr_node]
        # And i is in prev_mask ^ (1<<curr_node)
        
        # Optimization: Since n is small, we can just recompute or store parents.
        # Storing parents is cleaner. Let's redo with parent storage.
        pass
        
    # Let's restart the DP with parent tracking for correctness and simplicity of extraction.
    
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    
    for i in range(n):
        dp[1 << i][i] = dist_matrix[start, nodes[i]]
        
    for mask in range(1, 1 << n):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == INF:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                cost = dp[mask][i] + dist_matrix[nodes[i], nodes[j]]
                if cost < dp[new_mask][j]:
                    dp[new_mask][j] = cost
                    parent[new_mask][j] = i
                    
    full_mask = (1 << n) - 1
    best_cost = INF
    best_last_node_idx = -1
    for i in range(n):
        total_cost = dp[full_mask][i] + dist_matrix[nodes[i], end]
        if total_cost < best_cost:
            best_cost = total_cost
            best_last_node_idx = i
            
    # Backtrack to find the first node
    curr_mask = full_mask
    curr_node = best_last_node_idx
    while parent[curr_mask][curr_node] != -1:
        curr_node = parent[curr_mask][curr_node]
        curr_mask = curr_mask ^ (1 << curr_node)
        
    # curr_node is the index in 'nodes' of the first visited node
    return nodes[curr_node]


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1 # Should not happen
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Use exact solver for small subsets to ensure optimal closing
    if n_unvisited <= 10:
        return _solve_small_tsp(current_node, unvisited_nodes, destination_node, distance_matrix)
        
    # For larger sets, use a lookahead heuristic
    # Score = d(current, j) + lambda * d(j, nearest_neighbor_in_unvisited_excl_j) + mu * d(j, destination)
    
    # To avoid O(N^2) for nearest neighbor calculation inside the loop, 
    # we can precompute distances from all unvisited nodes to all other unvisited nodes?
    # That's O(N^2) which is fine for N=50.
    
    # Let's stick to a simpler lookahead: 
    # d(current, j) + d(j, argmin_k d(j, k)) where k != j
    
    best_node = -1
    best_score = float('inf')
    
    # Precompute distances from current to all unvisited
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate j, we need the distance to its nearest neighbor in unvisited \ {j}
    # We can compute this on the fly or precompute a "nearest neighbor distance" array for the subgraph.
    
    # Efficient approach:
    # For each j in unvisited_nodes:
    #   Find min distance from j to any other node in unvisited_nodes.
    
    # Since N is small (50), O(N^2) is acceptable.
    
    for idx, j in enumerate(unvisited_nodes):
        # Distance from current to j
        cost_to_j = dists_from_current[idx]
        
        # Lookahead: distance from j to its nearest neighbor in unvisited_nodes (excluding j)
        # If this is the last node, we should consider distance to destination.
        # But we only use this path for n_unvisited > 10, so there are at least 2 other nodes.
        
        # Find nearest neighbor of j among unvisited_nodes
        # Exclude j itself
        min_nn_dist = float('inf')
        # We can vectorize this part if we want, but loop is fine for N=50
        for k in unvisited_nodes:
            if k == j:
                continue
            d = distance_matrix[j, k]
            if d < min_nn_dist:
                min_nn_dist = d
                
        # Heuristic score
        # Weighted sum: immediate cost + future cost estimate
        # A common improvement is to add a small penalty for distance to destination
        # to guide the tour back, especially as we get closer to the end.
        
        # Let's use a simple score: cost_to_j + min_nn_dist
        # We can tune the weight of min_nn_dist.
        score = cost_to_j + 0.5 * min_nn_dist
        
        # Optional: Add destination pull
        # As n_unvisited decreases, we should care more about returning to destination.
        # But for n_unvisited > 10, the pull is weak.
        # Let's add a small constant weight to destination distance.
        score += 0.1 * distance_matrix[j, destination_node]
        
        if score < best_score:
            best_score = score
            best_node = j
            
    return int(best_node)
# EVOLVE-BLOCK-END
