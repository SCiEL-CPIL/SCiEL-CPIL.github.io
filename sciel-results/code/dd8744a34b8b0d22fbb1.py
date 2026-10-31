# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT_LIMIT = 145.0  # Leave 5s margin for 150s limit

def _is_timeout():
    return (time.time() - _START_TIME) > _TIMEOUT_LIMIT

def _mst_weight(nodes, distance_matrix):
    """
    Calculate the weight of the Minimum Spanning Tree for the given set of nodes.
    Uses Prim's algorithm with a simple loop, which is sufficient for small N.
    """
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Standard Prim's algorithm on the subset of nodes
    # dist_to_mst[i] is the min distance from node i to the current MST
    dist_to_mst = [float('inf')] * n
    dist_to_mst[0] = 0.0
    visited = [False] * n
    
    total_weight = 0.0
    
    for _ in range(n):
        # Find unvisited node with min dist_to_mst
        min_val = float('inf')
        u = -1
        for i in range(n):
            if not visited[i] and dist_to_mst[i] < min_val:
                min_val = dist_to_mst[i]
                u = i
        
        if u == -1:
            break # Should not happen in complete graph
            
        visited[u] = True
        total_weight += min_val
        
        # Update neighbors
        u_global = nodes[u]
        for v in range(n):
            if not visited[v]:
                v_global = nodes[v]
                d = distance_matrix[u_global, v_global]
                if d < dist_to_mst[v]:
                    dist_to_mst[v] = d
                    
    return total_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in a TSP tour using a hybrid MST-aware and minimax heuristic.
    """
    if _is_timeout():
        # Fallback to nearest neighbor if we are running out of time
        if len(unvisited_nodes) == 0:
            return int(current_node)
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = unvisited_nodes
        # Cost if we go to n1 first: current -> n1 -> n2 -> destination
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        # Cost if we go to n2 first: current -> n2 -> n1 -> destination
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return int(n1 if cost1 <= cost2 else n2)

    # For n_unvisited > 2, use hybrid heuristic
    # Score(u) = d(current, u) + alpha * MST(U \ {u}) + beta * max_{v in U\{u}} d(u, v)
    
    best_score = float('inf')
    best_node = int(unvisited_nodes[0])
    
    # Pre-compute distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Pre-compute distance matrix between unvisited nodes for the minimax term
    # This is O(N^2) which is acceptable for N=50
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Compute max distance from each u to any other v in unvisited (excluding itself)
    # Set diagonal to 0 so it doesn't affect max (distances are non-negative)
    d_unvisited_for_max = d_unvisited.copy()
    np.fill_diagonal(d_unvisited_for_max, 0)
    
    max_dist_to_other = np.max(d_unvisited_for_max, axis=1)
    
    # Weights for the hybrid score
    alpha = 1.0  # Weight for MST term
    beta = 0.1   # Weight for minimax term (inspiration 1)
    
    # Iterate over each candidate
    for i, u in enumerate(unvisited_nodes):
        # Immediate cost
        immediate_cost = dist_from_current[i]
        
        # Structural cost 1: MST of remaining nodes
        remaining_mask = np.ones(n_unvisited, dtype=bool)
        remaining_mask[i] = False
        remaining_nodes = unvisited_nodes[remaining_mask]
        
        # Calculate MST weight of remaining_nodes
        # If remaining is empty, MST weight is 0
        if len(remaining_nodes) == 0:
            mst_w = 0.0
        else:
            mst_w = _mst_weight(remaining_nodes, distance_matrix)
            
        # Structural cost 2: Minimax term
        minimax_cost = max_dist_to_other[i]
        
        total_score = immediate_cost + alpha * mst_w + beta * minimax_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = int(u)
            
    return best_node

# EVOLVE-BLOCK-END
