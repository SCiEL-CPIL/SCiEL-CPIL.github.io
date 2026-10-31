"""
Traveling Salesman constructive heuristic using MST-based residual cost estimation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 145.0
_START_TIME = time.time()


def _mst_weight(unvisited_indices, distance_matrix):
    """
    Compute the weight of the MST over the given unvisited indices using Prim's algorithm.
    This provides a lower bound on the remaining tour length through these nodes.
    """
    n = len(unvisited_indices)
    if n <= 1:
        return 0.0
    
    # Use Prim's algorithm with a priority queue approach (simplified for small n)
    # For n up to 50, O(n^2) Prim is fine
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    
    total_weight = 0.0
    
    for _ in range(n):
        # Find the vertex with minimum distance not yet in MST
        # For small n, linear scan is fine
        u = -1
        best = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best:
                best = min_dist[i]
                u = i
        
        if u == -1:
            break
            
        in_mst[u] = True
        total_weight += min_dist[u]
        
        # Update distances to remaining vertices
        for v in range(n):
            if not in_mst[v]:
                d = distance_matrix[unvisited_indices[u], unvisited_indices[v]]
                if d < min_dist[v]:
                    min_dist[v] = d
    
    return total_weight


def _mst_weight_fast(unvisited_indices, distance_matrix):
    """
    Vectorized Prim's algorithm for MST weight computation.
    """
    n = len(unvisited_indices)
    if n <= 1:
        return 0.0
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    
    total_weight = 0.0
    
    for _ in range(n):
        # Find min among not-in-MST
        candidates = ~in_mst
        if not np.any(candidates):
            break
        candidate_indices = np.where(candidates)[0]
        u = candidate_indices[np.argmin(min_dist[candidate_indices])]
        
        in_mst[u] = True
        total_weight += min_dist[u]
        
        # Vectorized update
        if not np.all(in_mst):
            dists = distance_matrix[unvisited_indices[u], unvisited_indices]
            mask = ~in_mst
            min_dist[mask] = np.minimum(min_dist[mask], dists[mask])
    
    return total_weight


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick next city using MST-based residual cost estimation combined with 
    nearest neighbor and destination bias.
    
    The score for each candidate i is:
      d(current, i) + alpha * (MST_weight(unvisited \ {i}) - MST_weight(unvisited))
    
    This encourages picking nodes that, when removed, reduce the MST weight the most,
    which correlates with good tour structure.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited <= 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        return int(unvisited_nodes[0])
    
    d_curr = distance_matrix[current_node, unvisited_nodes]
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Base score: nearest neighbor distance
    base_score = d_curr.copy()
    
    # Adaptive destination bias
    if n_unvisited <= 5:
        w_dest = 0.5
    elif n_unvisited <= 15:
        w_dest = 0.3
    else:
        w_dest = 0.15
    
    score = base_score + w_dest * d_dest
    
    # MST-based improvement: for larger instances, use MST residual cost
    if n_unvisited <= 30:
        # Compute MST weight for the full unvisited set
        mst_full = _mst_weight_fast(unvisited_nodes, distance_matrix)
        
        # For each candidate, estimate the MST weight if we remove it
        # This is expensive, so we use a cheaper approximation:
        # The change in MST weight when removing node i is related to 
        # the maximum edge in the MST incident to i.
        # We approximate by: for each i, the cost increase is roughly
        # the max distance from i to its two nearest neighbors in the unvisited set.
        
        # Compute for each node its two nearest neighbors' distances
        # This is O(n^2) which is fine for n <= 30
        nn_costs = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            dists = distance_matrix[unvisited_nodes[i], unvisited_nodes].copy()
            dists[i] = np.inf
            sorted_d = np.sort(dists)
            # Sum of two smallest distances (approximation of MST contribution)
            nn_costs[i] = sorted_d[0] + sorted_d[1]
        
        # Normalize nn_costs
        min_nn = np.min(nn_costs)
        max_nn = np.max(nn_costs)
        if max_nn > min_nn:
            nn_norm = (nn_costs - min_nn) / (max_nn - min_nn)
        else:
            nn_norm = np.zeros(n_unvisited)
        
        # Prefer nodes with LOW nn_cost (well-connected nodes)
        # Add a penalty for isolated nodes
        w_mst = 0.2
        score = score + w_mst * nn_norm
    
    # Additional: regret-based adjustment for medium-sized sets
    if 5 < n_unvisited <= 25:
        # Compute 2-step lookahead
        lookahead_costs = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            dists_from_i = distance_matrix[unvisited_nodes[i], unvisited_nodes].copy()
            dists_from_i[i] = np.inf
            min_next = np.min(dists_from_i)
            lookahead_costs[i] = d_curr[i] + min_next
        
        min_la = np.min(lookahead_costs)
        max_la = np.max(lookahead_costs)
        if max_la > min_la:
            la_norm = (lookahead_costs - min_la) / (max_la - min_la)
        else:
            la_norm = np.zeros(n_unvisited)
        
        w_la = 0.15
        score = score + w_la * la_norm
    
    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
