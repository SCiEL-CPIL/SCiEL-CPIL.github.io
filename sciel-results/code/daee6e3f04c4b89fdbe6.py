import numpy as np
import itertools
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return current_node 

    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Time guard: track elapsed time
    start_time = time.time()
    deadline = start_time + 1.5  # 1.5 second budget per call

    # If the number of remaining nodes is small, solve the sub-problem exactly
    if n_unvisited <= 7:
        best_cost = np.inf
        best_next = int(unvisited_nodes[0])
        
        for perm in itertools.permutations(unvisited_nodes):
            cost = distance_matrix[current_node, perm[0]]
            for i in range(n_unvisited - 1):
                cost += distance_matrix[perm[i], perm[i+1]]
            cost += distance_matrix[perm[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_next = int(perm[0])
                
        return best_next

    # For larger sets, use enhanced lookahead with destination awareness
    # Get sub-matrix of distances between unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Set diagonal to infinity so we don't pick the node itself
    sub_dist_no_diag = sub_dist.copy()
    np.fill_diagonal(sub_dist_no_diag, np.inf)
    
    # Distance from each unvisited node to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For each node, find its top-3 nearest neighbors (excluding itself)
    sorted_dists = np.sort(sub_dist_no_diag, axis=1)
    nn1_dist = sorted_dists[:, 0]
    nn2_dist = sorted_dists[:, 1]
    nn3_dist = sorted_dists[:, 2]
    
    # Get indices of nearest neighbors
    nn1_idx = np.argmin(sub_dist_no_diag, axis=1)
    
    # For 2nd nearest, need to exclude nn1
    # Create a copy with nn1 column set to inf for each row
    sub_dist_excl_nn1 = sub_dist_no_diag.copy()
    for i in range(n_unvisited):
        sub_dist_excl_nn1[i, nn1_idx[i]] = np.inf
    nn2_idx = np.argmin(sub_dist_excl_nn1, axis=1)
    
    # For 3rd nearest, exclude nn1 and nn2
    sub_dist_excl_nn12 = sub_dist_no_diag.copy()
    for i in range(n_unvisited):
        sub_dist_excl_nn12[i, nn1_idx[i]] = np.inf
        sub_dist_excl_nn12[i, nn2_idx[i]] = np.inf
    nn3_idx = np.argmin(sub_dist_excl_nn12, axis=1)
    
    # Distance from current_node to each unvisited node
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # For each candidate v (local index i), compute score using 4-step lookahead
    # score[i] = d(current, v) + min over j != i of [d(v, j) + cont3(j, excluding i)]
    # where cont3(j, excluding i) approximates the cost of 3 more steps from j
    # cont3(j, excl) = min over k != j,excl of [d(j,k) + cont2(k, excluding j,excl)]
    # cont2(k, excl_set) = min over m != k,excl_set of [d(k,m) + d(m, dest)]
    #   or if no more nodes, just d(k, dest)
    
    best_scores = np.full(n_unvisited, np.inf)
    
    for i in range(n_unvisited):
        # Check time
        if time.time() > deadline:
            break
        
        # For candidate v = unvisited_nodes[i]
        # For each w = unvisited_nodes[j] where j != i:
        #   cost = d(v, j) + best_3step_cont(j, excluding i)
        
        j_range = np.arange(n_unvisited)
        mask = j_range != i
        
        d_v_j = sub_dist[i, mask]  # distances from v to all other unvisited nodes
        
        # For each j in mask, compute the best 3-step continuation cost
        # This is: min over k != j, i of [d(j,k) + best_2step_cont(k, excluding j, i)]
        # where best_2step_cont(k, excl) = min over m != k, excl of [d(k,m) + d(m,dest)]
        #   or if only one node left after k, just d(k,dest)
        
        cont3_costs = np.zeros(np.sum(mask))
        
        for pos, j in enumerate(j_range[mask]):
            # For node j, find best k != j, i
            k_candidates = np.arange(n_unvisited)
            k_mask = (k_candidates != j) & (k_candidates != i)
            
            if np.sum(k_mask) == 0:
                # No more nodes after j, so cost is just d(j, dest)
                cont3_costs[pos] = dist_to_dest[j]
                continue
            
            d_j_k = sub_dist[j, k_mask]
            
            # For each k, compute best 2-step continuation
            # best_2step_cont(k, excluding j, i) = min over m != k, j, i of [d(k,m) + d(m,dest)]
            #   or if no m left, just d(k, dest)
            
            m_candidates = np.arange(n_unvisited)
            m_mask = (m_candidates != j) & (m_candidates != i) & (m_candidates != k_candidates[0])  # This is wrong, need per-k
            
            # Actually, for each k, we need m != k, j, i
            # Let's vectorize this differently
            # For each k in k_mask positions:
            k_indices = k_candidates[k_mask]
            
            # For each k, find best m != k, j, i
            # Create a matrix where for each k, we have d(k,m) + d(m,dest) for all valid m
            # m must be != k, j, i
            
            cont2_costs = np.zeros(len(k_indices))
            for ki, k in enumerate(k_indices):
                m_valid = (np.arange(n_unvisited) != k) & (np.arange(n_unvisited) != j) & (np.arange(n_unvisited) != i)
                if np.sum(m_valid) == 0:
                    cont2_costs[ki] = dist_to_dest[k]
                else:
                    d_k_m = sub_dist[k, m_valid]
                    d_m_dest = dist_to_dest[m_valid]
                    cont2_costs[ki] = np.min(d_k_m + d_m_dest)
            
            # Now for each k, total cost is d(j,k) + cont2_cost(k)
            total_k_costs = d_j_k + cont2_costs
            cont3_costs[pos] = np.min(total_k_costs)
        
        total_costs = d_v_j + cont3_costs
        best_3step = np.min(total_costs)
        
        # Add destination penalty to encourage moving toward destination
        dest_penalty = 0.15 * dist_to_dest[i]
        
        best_scores[i] = dist_from_current[i] + best_3step + dest_penalty
    
    # If time ran out, fall back to simpler heuristic
    if time.time() > deadline:
        # Fallback: 2-step lookahead with destination bias
        scores = dist_from_current + nn1_dist + 0.15 * dist_to_dest
        best_local_idx = np.argmin(scores)
        return int(unvisited_nodes[best_local_idx])
    
    best_local_idx = np.argmin(best_scores)
    return int(unvisited_nodes[best_local_idx])
