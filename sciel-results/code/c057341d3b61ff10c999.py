"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 145.0  # seconds; safety margin before 150s kill
_START_TIME = time.time()


def _fast_nearest_neighbor(current_node, unvisited_nodes, distance_matrix):
    """Fast fallback: pick the nearest unvisited node to current."""
    if len(unvisited_nodes) <= 1:
        return int(unvisited_nodes[0])
    dists = distance_matrix[current_node, unvisited_nodes]
    return int(unvisited_nodes[np.argmin(dists)])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick next city using 2-step lookahead with global connectivity awareness.
    
    Structural change: Instead of static feature scoring, we perform a short-horizon
    search. For each candidate, we estimate the 2-step cost (current -> candidate -> 
    best next) and combine with a global MST-like connectivity measure. This captures
    local optimality (avoid dead-end choices) while maintaining global tour structure.
    """
    # Deadline guard: if we're close to the limit, fall back to fast NN
    if time.time() > _DEADLINE:
        return _fast_nearest_neighbor(current_node, unvisited_nodes, distance_matrix)

    n_unvisited = len(unvisited_nodes)
    if n_unvisited <= 1:
        return int(unvisited_nodes[0])
    
    # For very small sets, use exhaustive 2-step lookahead
    if n_unvisited <= 3:
        # Try all candidates, pick the one with best 2-step cost
        best_cost = np.inf
        best_idx = 0
        for i in range(n_unvisited):
            node_i = unvisited_nodes[i]
            cost_1 = distance_matrix[current_node, node_i]
            if n_unvisited == 2:
                # Must go to the other node
                other = unvisited_nodes[1 - i]
                cost_2 = distance_matrix[node_i, other]
                total = cost_1 + cost_2
            else:
                # For 3 nodes, try both continuations
                min_cost_2 = np.inf
                for j in range(n_unvisited):
                    if j != i:
                        node_j = unvisited_nodes[j]
                        cost_2 = distance_matrix[node_i, node_j]
                        min_cost_2 = min(min_cost_2, cost_2)
                total = cost_1 + min_cost_2
            if total < best_cost:
                best_cost = total
                best_idx = i
        return int(unvisited_nodes[best_idx])

    # Distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    # Distances from destination to all unvisited
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]

    # Adaptive weights based on remaining nodes
    frac_remaining = n_unvisited / 50.0  # Assume ~50 cities total
    if n_unvisited <= 5:
        dest_weight = 0.7
        lookahead_weight = 0.3
    elif n_unvisited <= 10:
        dest_weight = 0.4
        lookahead_weight = 0.6
    elif n_unvisited <= 20:
        dest_weight = 0.2
        lookahead_weight = 0.8
    else:
        dest_weight = 0.1
        lookahead_weight = 1.0

    # 2-step lookahead: for each candidate i, estimate min_{j != i} dist(i, j)
    # This is the best possible next step after choosing i
    # We compute this efficiently using the submatrix
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each candidate i, the best 2nd step is min_{j != i} sub_dist[i, j]
    # We can compute this by finding the minimum of each row, but excluding the diagonal
    # Copy the matrix and set diagonal to inf, then take row mins
    sub_dist_no_diag = sub_dist.copy()
    np.fill_diagonal(sub_dist_no_diag, np.inf)
    best_second_step = np.min(sub_dist_no_diag, axis=1)  # shape (n_unvisited,)
    
    # 2-step cost for candidate i: dist(current, i) + best_second_step[i]
    two_step_cost = dist_from_current + best_second_step
    
    # Global connectivity: estimate how well-connected the remaining graph is
    # Use a proxy: the average of the k-smallest distances from each node to others
    # This captures whether we're fragmenting the tour
    if n_unvisited <= 30:
        # For each node, compute the average of the 3 smallest distances to other unvisited nodes
        # This is a cheap measure of local connectivity
        k = min(3, n_unvisited - 1)
        if k > 0:
            # For each row, find the k smallest values (excluding diagonal)
            # Use argpartition for efficiency
            connectivity_scores = np.zeros(n_unvisited)
            for j in range(n_unvisited):
                row = sub_dist_no_diag[j]
                if k < len(row):
                    # Get the k smallest values
                    k_smallest = np.partition(row, k)[:k]
                    connectivity_scores[j] = np.mean(k_smallest)
                else:
                    connectivity_scores[j] = np.mean(row)
            
            # For candidate i, the connectivity of the remaining set (excluding i)
            # is approximately the max of connectivity_scores[j] for j != i
            # (worst-case isolated node in the remaining set)
            # Use the max excluding i as a proxy for fragmentation risk
            sorted_conn = np.sort(connectivity_scores)
            global_max_conn = sorted_conn[-1]
            second_max_conn = sorted_conn[-2] if n_unvisited > 1 else 0
            
            frag_costs = np.full(n_unvisited, global_max_conn)
            mask = (connectivity_scores == global_max_conn)
            frag_costs[mask] = second_max_conn
            
            # Normalize
            frag_range = frag_costs.max() - frag_costs.min()
            if frag_range > 1e-9:
                frag_norm = (frag_costs - frag_costs.min()) / frag_range
            else:
                frag_norm = np.zeros(n_unvisited)
        else:
            frag_norm = np.zeros(n_unvisited)
    else:
        frag_norm = np.zeros(n_unvisited)

    # Normalize components to [0, 1]
    def normalize(arr):
        rng = arr.max() - arr.min()
        if rng > 1e-9:
            return (arr - arr.min()) / rng
        return np.zeros_like(arr)
    
    two_step_norm = normalize(two_step_cost)
    dest_norm = normalize(dist_to_dest)

    # Weighted score: lower is better
    # Primary: 2-step lookahead (captures local dead-ends)
    # Secondary: destination bias (ensures we can close the tour)
    # Tertiary: fragmentation risk (maintains global connectivity)
    frag_weight = 0.15 if n_unvisited <= 30 else 0.0
    score = (lookahead_weight * two_step_norm + 
             dest_weight * dest_norm + 
             frag_weight * frag_norm)

    best_idx = np.argmin(score)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
