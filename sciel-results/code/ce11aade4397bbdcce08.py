"""TSP constructive heuristic: minimax-bottleneck centrality + destination awareness with per-instance calibration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 150s limit

# Per-instance state
_FIT_PARAMS = None  # (alpha, beta, gamma) best found by fitting
_current_dm_shape = None


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _build_tour(distance_matrix, start_node, alpha=0.0, beta=0.0, gamma=0.0):
    """Build a full TSP tour using the given parameters. Returns tour length."""
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(range(1, n))  # 0 is start
    
    while len(remaining) > 0:
        n_rem = len(remaining)
        
        if n_rem == 1:
            next_node = remaining[0]
            total_dist += distance_matrix[current, next_node]
            current = next_node
            break
        
        if n_rem == 2:
            n1, n2 = remaining[0], remaining[1]
            cost1 = distance_matrix[current, n1] + distance_matrix[n1, n2] + distance_matrix[n2, start_node]
            cost2 = distance_matrix[current, n2] + distance_matrix[n2, n1] + distance_matrix[n1, start_node]
            if cost1 <= cost2:
                next_node = n1
            else:
                next_node = n2
            total_dist += distance_matrix[current, next_node]
            current = next_node
            other = n2 if next_node == n1 else n1
            total_dist += distance_matrix[current, other]
            current = other
            break
        
        # Vectorized scoring for n_rem > 2
        d_cur = distance_matrix[current, remaining]
        scores = d_cur.copy()
        
        if alpha > 0:
            d_dest = distance_matrix[remaining, start_node]
            scores += alpha * d_dest
        
        if beta > 0:
            # Minimax centrality: penalize nodes that are far from the closest other unvisited node
            # This is the "bottleneck" component - we want to avoid isolated nodes
            sub_dist = distance_matrix[np.ix_(remaining, remaining)]
            np.fill_diagonal(sub_dist, np.inf)
            min_dist_to_other = np.min(sub_dist, axis=1)
            scores += beta * min_dist_to_other
        
        if gamma > 0:
            # Max distance penalty: penalize nodes whose farthest unvisited neighbor is very far
            # This directly targets the minimax objective - avoid creating long edges
            sub_dist2 = distance_matrix[np.ix_(remaining, remaining)]
            np.fill_diagonal(sub_dist2, np.inf)
            max_dist_to_other = np.max(sub_dist2, axis=1)
            # Normalize by the overall max to keep scale consistent
            overall_max = np.max(max_dist_to_other)
            if overall_max > 0:
                scores += gamma * (max_dist_to_other / overall_max)
        
        best_idx = np.argmin(scores)
        next_node = remaining[best_idx]
        total_dist += distance_matrix[current, next_node]
        remaining.pop(best_idx)
        current = next_node
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=4.0):
    """Fit alpha, beta, gamma by grid search on the full instance."""
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0)
    
    # Grid of parameters
    alpha_values = [0.0, 0.05, 0.10, 0.20]
    beta_values = [0.0, 0.25, 0.50, 0.75, 1.0]
    gamma_values = [0.0, 0.1, 0.2, 0.4]
    
    best_alpha = 0.0
    best_beta = 0.0
    best_gamma = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    deadline = elapsed_start + min(time_budget, max(0.5, _LIMIT - (time.time() - _start_time) - 5.0))
    
    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for beta in beta_values:
            if time.time() > deadline:
                break
            for gamma in gamma_values:
                if time.time() > deadline:
                    break
                tour_len = _build_tour(distance_matrix, start_node, alpha, beta, gamma)
                if tour_len < best_len:
                    best_len = tour_len
                    best_alpha = alpha
                    best_beta = beta
                    best_gamma = gamma
    
    _FIT_PARAMS = (best_alpha, best_beta, best_gamma)
    return (best_alpha, best_beta, best_gamma)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses minimax-bottleneck centrality: favors nodes that are well-connected to the
    rest of the unvisited set (low max-distance-to-others) while maintaining some
    nearest-neighbor behavior. Per-instance parameter fitting calibrates the balance.
    """
    global _FIT_PARAMS, _current_dm_shape
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        n1, n2 = int(unvisited_nodes[0]), int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, n1] + distance_matrix[n1, n2] + distance_matrix[n2, destination_node]
        cost2 = distance_matrix[current_node, n2] + distance_matrix[n2, n1] + distance_matrix[n1, destination_node]
        return n1 if cost1 <= cost2 else n2
    
    # Detect new instance
    dm_shape = distance_matrix.shape
    if dm_shape != _current_dm_shape:
        _current_dm_shape = dm_shape
        _FIT_PARAMS = None
    
    # Fit parameters if not yet done for this instance
    if _FIT_PARAMS is None:
        _fit_params(distance_matrix, destination_node)
    
    alpha, beta, gamma = _FIT_PARAMS if _FIT_PARAMS else (0.0, 0.0, 0.0)
    
    # Check deadline
    if _is_deadline_passed():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute scores
    d_cur = distance_matrix[current_node, unvisited_nodes]
    scores = d_cur.copy()
    
    if alpha > 0:
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * d_dest
    
    if beta > 0:
        # Minimax centrality: penalize isolated nodes (high min-distance-to-others)
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_dist_to_other = np.min(sub_dist, axis=1)
        scores += beta * min_dist_to_other
    
    if gamma > 0:
        # Max distance penalty: penalize nodes with very far unvisited neighbors
        sub_dist2 = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist2, np.inf)
        max_dist_to_other = np.max(sub_dist2, axis=1)
        overall_max = np.max(max_dist_to_other)
        if overall_max > 0:
            scores += gamma * (max_dist_to_other / overall_max)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
