import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    heuristic = np.ones((n, n)) * 1e-9
    
    # Precompute reachability
    # reachable[i, j] is True if j != i, j != 0, and d[i, j] + d[j, 0] <= max_len
    # Note: The ant starts at 0. It moves to j. The condition in the task is:
    # "moves to an unvisited node j only if the length so far plus d[i, j] plus d[j, 0] stays within max_len"
    # The heuristic is computed once. We don't know the "length so far" for each ant at each step.
    # However, the heuristic is static. The standard ACO heuristic $\eta_{ij}$ is usually static.
    # The feasibility check is dynamic.
    # The current code uses `distance_matrix[i, k] + distance_matrix[k, 0] <= max_len` for reachability.
    # This assumes the path is 0 -> ... -> i -> k -> 0.
    # Since the heuristic is static, we can assume the "worst case" or "average case" reachability?
    # The current code assumes if `d[i,k] + d[k,0] <= max_len`, then k is a valid candidate from i.
    # This is a necessary condition for k to be visitable from i and return to 0, assuming we are at i.
    # It ignores the path taken to get to i. But since the heuristic is static, this is the best we can do.
    
    reachable = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for k in range(n):
            if k != i and k != 0:
                if distance_matrix[i, k] + distance_matrix[k, 0] <= max_len:
                    reachable[i, k] = True

    eps = 1e-9
    
    # Vectorized calculation of heuristic
    for i in range(n):
        # Get indices of reachable nodes from i
        reachable_nodes = np.where(reachable[i, :])[0]
        if len(reachable_nodes) == 0:
            continue
            
        # Get prizes for these nodes
        node_prizes = prizes[reachable_nodes]
        
        # Find top 2 prizes and their indices in the original array
        # We need to be careful with ties, but np.argsort handles it.
        # We want the max prize and the second max prize among reachable nodes.
        
        # Sort prizes in descending order
        sorted_indices = np.argsort(node_prizes)[::-1]
        
        if len(sorted_indices) == 1:
            max1_prize = node_prizes[sorted_indices[0]]
            max1_node = reachable_nodes[sorted_indices[0]]
            max2_prize = 0.0 # No alternative
            # For the single node, max_alt is 0
        else:
            max1_prize = node_prizes[sorted_indices[0]]
            max1_node = reachable_nodes[sorted_indices[0]]
            max2_prize = node_prizes[sorted_indices[1]]
            
        # Initialize row for heuristic
        # We will fill heuristic[i, j] for j in reachable_nodes
        
        # For all j in reachable_nodes:
        # M_j = max1_prize if j != max1_node else max2_prize
        
        # Create a temporary array for max_alt for these nodes
        # Default to max1_prize
        max_alt_values = np.full(len(reachable_nodes), max1_prize)
        
        # Find the index of max1_node in reachable_nodes
        # Since max1_node is in reachable_nodes, we can find its position
        # Note: if there are ties, sorted_indices[0] is the first max.
        # If j is that specific node, its alt is max2.
        # If j is another node with the same max prize, its alt is also max1 (which is the same value).
        # So we just need to check if the node ID matches.
        
        # Get the index of max1_node in the reachable_nodes array
        # np.where(reachable_nodes == max1_node)[0][0]
        idx_max1 = np.where(reachable_nodes == max1_node)[0][0]
        max_alt_values[idx_max1] = max2_prize
        
        # Now calculate heuristic for each j
        for idx, j in enumerate(reachable_nodes):
            # j is reachable from i
            # heuristic = (prize[j] / (dist[i,j] + eps)) / (max_alt_values[idx] + eps)
            dist_ij = distance_matrix[i, j]
            prize_j = prizes[j]
            
            # If prize is 0, heuristic is 0 (or 1e-9)
            if prize_j <= 0:
                continue
                
            val = (prize_j / (dist_ij + eps)) / (max_alt_values[idx] + eps)
            
            if val > 0 and np.isfinite(val):
                heuristic[i, j] = val
            else:
                heuristic[i, j] = 1e-9

    # Ensure non-finite values become 1e-9
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
