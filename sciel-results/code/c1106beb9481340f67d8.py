# EVOLVE-BLOCK-START
import numpy as np

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node in a TSP tour using a heuristic that combines
    immediate distance, destination distance, and a connectivity-based residual estimate
    derived from nearest-neighbor structures (proxy for MST connectivity).
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Pre-compute nearest neighbor for each node in unvisited_nodes
    # This serves as a proxy for local connectivity costs.
    # nn_idx[i] is the index in unvisited_nodes of the nearest neighbor to unvisited_nodes[i]
    # nn_dist[i] is the distance to that nearest neighbor
    
    nn_idx = np.zeros(n_unvisited, dtype=int)
    nn_dist = np.zeros(n_unvisited, dtype=float)
    
    # Efficiently compute nearest neighbors using NumPy
    # Create a mask to ignore self-distances
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(sub_dist, np.inf)
    
    nn_idx = np.argmin(sub_dist, axis=1)
    nn_dist = sub_dist[np.arange(n_unvisited), nn_idx]
    
    # Sum of nearest neighbor distances for the current set
    total_snn = np.sum(nn_dist)
    
    best_score = float('inf')
    best_node = -1
    
    # We will iterate through candidates.
    # To calculate SNN(V \ {u}), we need to adjust total_snn.
    # 1. Remove the contribution of u: -nn_dist[u]
    # 2. For any v where nn_idx[v] == u, v needs a new nearest neighbor in V \ {u, v}.
    #    We update nn_dist[v] and adjust the sum.
    
    # Pre-identify which nodes point to each candidate to speed up updates?
    # Or just do it on the fly. N is small (<=50).
    
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # Immediate costs
        immediate_dist = distance_matrix[current_node, candidate]
        dest_dist = distance_matrix[candidate, destination_node]
        
        # Calculate Residual Estimate
        # We estimate the cost to visit the remaining nodes V' = V \ {candidate}
        # and connect candidate to V' and V' to destination.
        
        # 1. Cost to enter V' from candidate
        if n_unvisited == 2:
            enter_cost = distance_matrix[candidate, unvisited_nodes[1-i]]
        else:
            # Find nearest neighbor of candidate in V \ {candidate}
            # We can use the precomputed nn, but if nn_idx[i] is valid, it's the nearest in V.
            # Since we remove candidate, we need nearest in V \ {candidate}.
            # nn_idx[i] is the nearest in V \ {candidate} already? 
            # Yes, because sub_dist diagonal was inf, so nn_idx[i] is always != i.
            enter_cost = nn_dist[i]
            
        # 2. Cost to visit V' internally
        # Approximation: 0.5 * SNN(V')
        # SNN(V') = SNN(V) - nn_dist[i] + sum of adjustments for nodes v where nn_idx[v] == i
        
        snn_v_prime = total_snn - nn_dist[i]
        
        # Find nodes v in V \ {i} such that nn_idx[v] == i
        # These nodes need their nearest neighbor recalculated in V \ {i, v}
        affected_nodes = np.where(nn_idx == i)[0]
        # Exclude i itself if it appeared (it shouldn't, as nn_idx[i] != i)
        affected_nodes = affected_nodes[affected_nodes != i]
        
        for v_idx in affected_nodes:
            # v_idx is index in unvisited_nodes
            v_node = unvisited_nodes[v_idx]
            # Find new nearest neighbor for v_node in V \ {i, v_idx}
            # Create a temporary distance vector
            dists = sub_dist[v_idx].copy()
            dists[i] = np.inf
            dists[v_idx] = np.inf
            new_nn_idx = np.argmin(dists)
            new_nn_dist = dists[new_nn_idx]
            
            # Update snn_v_prime
            snn_v_prime -= nn_dist[v_idx]
            snn_v_prime += new_nn_dist
            
            # Note: We don't update nn_idx or nn_dist in place for other candidates,
            # so we must be careful. But since we recompute from scratch for each candidate
            # based on the ORIGINAL nn_idx/nn_dist, this is safe.
            # However, if multiple nodes point to i, they are independent updates to the sum.
            
        residual_internal = 0.5 * snn_v_prime
        
        # 3. Cost to exit V' to destination
        # Approximation: min distance from destination to any node in V'
        if n_unvisited == 2:
            exit_cost = distance_matrix[destination_node, unvisited_nodes[1-i]]
        else:
            # We need min dist from dest to V \ {i}
            dists_dest = distance_matrix[destination_node, unvisited_nodes]
            dists_dest[i] = np.inf
            exit_cost = np.min(dists_dest)
            
        total_score = immediate_dist + dest_dist + enter_cost + residual_internal + exit_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
            
    return int(best_node)
# EVOLVE-BLOCK-END
