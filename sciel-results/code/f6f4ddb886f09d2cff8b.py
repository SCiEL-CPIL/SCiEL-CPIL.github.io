"""Combined TSP constructive heuristic: per-instance parameter fitting + 2-step lookahead + isolation penalty."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 150s limit

# Per-instance state
_FIT_PARAMS = None  # (alpha, gamma, delta) best found by fitting
_instance_id = -1
_current_dm_shape = None


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _build_tour(distance_matrix, start_node, alpha=0.0, gamma=0.0, delta=0.0):
    """Build a full TSP tour using the given parameters. Returns tour length."""
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = np.arange(1, n)  # 0 is start
    visited_mask = np.zeros(n, dtype=bool)
    visited_mask[0] = True
    
    while len(remaining) > 0:
        n_rem = len(remaining)
        
        if n_rem == 1:
            next_node = int(remaining[0])
            total_dist += distance_matrix[current, next_node]
            current = next_node
            break
        
        if n_rem == 2:
            n1, n2 = int(remaining[0]), int(remaining[1])
            cost1 = distance_matrix[current, n1] + distance_matrix[n1, n2] + distance_matrix[n2, start_node]
            cost2 = distance_matrix[current, n2] + distance_matrix[n2, n1] + distance_matrix[n1, start_node]
            if cost1 <= cost2:
                next_node = n1
            else:
                next_node = n2
            total_dist += distance_matrix[current, next_node]
            current = next_node
            # Visit the other one
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
        
        if gamma > 0 and n_rem > 2:
            # For each candidate c, compute min distance from c to other remaining nodes
            sub_dist = distance_matrix[np.ix_(remaining, remaining)]
            np.fill_diagonal(sub_dist, np.inf)
            min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and n_rem > 2:
            # Isolation penalty: penalize candidates whose distance to dest
            # is much larger than the minimum among remaining
            d_dest_all = distance_matrix[remaining, start_node]
            min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        best_idx = np.argmin(scores)
        next_node = int(remaining[best_idx])
        total_dist += distance_matrix[current, next_node]
        visited_mask[next_node] = True
        remaining = np.delete(remaining, best_idx)
        current = next_node
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, gamma, delta by grid search on the full instance."""
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0)
    
    # Grid of parameters - smaller for speed
    alpha_values = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
    gamma_values = [0.0, 0.05, 0.10, 0.20]
    delta_values = [0.0, 0.05, 0.10]
    
    best_alpha = 0.0
    best_gamma = 0.0
    best_delta = 0.0
    best_len = float('inf')
    
    elapsed_start = time.time()
    deadline = elapsed_start + min(time_budget, max(0.5, _LIMIT - (time.time() - _start_time) - 5.0))
    
    for alpha in alpha_values:
        if time.time() > deadline:
            break
        for gamma in gamma_values:
            if time.time() > deadline:
                break
            for delta in delta_values:
                if time.time() > deadline:
                    break
                tour_len = _build_tour(distance_matrix, start_node, alpha, gamma, delta)
                if tour_len < best_len:
                    best_len = tour_len
                    best_alpha = alpha
                    best_gamma = gamma
                    best_delta = delta
    
    _FIT_PARAMS = (best_alpha, best_gamma, best_delta)
    return (best_alpha, best_gamma, best_delta)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Uses per-instance parameter fitting with 2-step lookahead and isolation penalty.
    """
    global _FIT_PARAMS, _instance_id, _current_dm_shape
    
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
        _instance_id += 1
        _current_dm_shape = dm_shape
        _FIT_PARAMS = None
    
    # Fit parameters if not yet done for this instance
    if _FIT_PARAMS is None:
        _fit_params(distance_matrix, destination_node)
    
    alpha, gamma, delta = _FIT_PARAMS if _FIT_PARAMS else (0.0, 0.0, 0.0)
    
    # Check deadline
    if _is_deadline_passed():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute scores
    d_cur = distance_matrix[current_node, unvisited_nodes]
    scores = d_cur.copy()
    
    if alpha > 0:
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * d_dest
    
    if gamma > 0 and n_unvisited > 2:
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += gamma * min_next
    
    if delta > 0 and n_unvisited > 2:
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = np.min(d_dest_all)
        scores += delta * (d_dest_all - min_d_dest)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
