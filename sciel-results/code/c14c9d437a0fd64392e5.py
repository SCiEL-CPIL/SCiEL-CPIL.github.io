"""Combined TSP constructive heuristic: model-and-optimize with 2-step lookahead.

Strategy:
  1. For each new instance (detected by unvisited count being high), fit parameters
     (alpha, gamma, delta) via a small grid search on the full instance.
  2. Use a vectorized 2-step lookahead score for efficiency:
     score(c) = d(cur,c) + alpha*d(c,dest) + gamma*min_{u in rem\\{c}} d(c,u)
                + delta * (d(c,dest) - min_u d(u,dest))  [isolation penalty]
  3. For small remaining sets (<=5), use exact brute force to optimize the tail.
  4. Deadline guard: if time is running out, fall back to nearest neighbor.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import permutations

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

# Per-instance state
_current_instance_id = -1
_FIT_PARAMS = None  # (alpha, gamma, delta) best found by fitting
_instance_counter = 0


def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT


def _build_tour(distance_matrix, start_node, alpha=0.0, gamma=0.0, delta=0.0):
    """Build a full TSP tour using the given parameters. Returns tour length."""
    n = distance_matrix.shape[0]
    current = start_node
    total_dist = 0.0
    remaining = list(np.arange(1, n))  # 0 is start
    
    while len(remaining) > 0:
        n_unvisited = len(remaining)
        
        # If only 1 node left, return it
        if n_unvisited == 1:
            next_node = int(remaining[0])
            total_dist += distance_matrix[current, next_node]
            current = next_node
            remaining = []
            break
        
        # If few nodes left, use exact brute force to optimize the tail
        if n_unvisited <= 5:
            nodes = list(remaining)
            best_cost = np.inf
            best_first_node = nodes[0]
            
            for perm in permutations(nodes):
                # Cost: current -> perm[0] -> ... -> perm[-1] -> destination
                cost = distance_matrix[current, perm[0]]
                for i in range(len(perm) - 1):
                    cost += distance_matrix[perm[i], perm[i+1]]
                cost += distance_matrix[perm[-1], start_node]
                
                if cost < best_cost:
                    best_cost = cost
                    best_first_node = perm[0]
            
            next_node = int(best_first_node)
            total_dist += distance_matrix[current, next_node]
            current = next_node
            remaining.remove(next_node)
            continue
        
        # Heuristic for larger sets: 2-step lookahead
        # 1. Distance from current to all unvisited
        dist_curr = distance_matrix[current, remaining]
        
        scores = dist_curr.copy()
        
        if alpha > 0:
            d_dest = distance_matrix[remaining, start_node]
            scores += alpha * d_dest
        
        if gamma > 0 and n_unvisited > 2:
            # For each candidate c, compute min distance from c to other remaining nodes
            # Vectorized: submatrix of remaining x remaining
            sub_dist = distance_matrix[np.ix_(remaining, remaining)]
            # Set diagonal to inf so we don't pick self
            np.fill_diagonal(sub_dist, np.inf)
            min_next = np.min(sub_dist, axis=1)
            scores += gamma * min_next
        
        if delta > 0 and n_unvisited > 2:
            # Isolation penalty: penalize candidates whose distance to dest
            # is much larger than the minimum among remaining
            d_dest_all = distance_matrix[remaining, start_node]
            min_d_dest = np.min(d_dest_all)
            scores += delta * (d_dest_all - min_d_dest)
        
        best_idx = np.argmin(scores)
        next_node = int(remaining[best_idx])
        total_dist += distance_matrix[current, next_node]
        current = next_node
        remaining.remove(next_node)
    
    # Close the tour
    total_dist += distance_matrix[current, start_node]
    return total_dist


def _fit_params(distance_matrix, start_node, time_budget=3.0):
    """Fit alpha, gamma, delta by grid search on the full instance."""
    global _FIT_PARAMS
    
    if _is_deadline_passed():
        return (0.0, 0.0, 0.0)
    
    # Grid of parameters
    alpha_values = [0.0, 0.03, 0.07, 0.12, 0.2, 0.35]
    gamma_values = [0.0, 0.05, 0.1, 0.15, 0.25]
    delta_values = [0.0, 0.05, 0.15]
    
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
    
    Uses model-and-optimize: fits parameters per instance via grid search,
    then uses them for all subsequent steps of that instance.
    """
    global _current_instance_id, _FIT_PARAMS, _instance_counter
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # If few nodes left, use exact brute force to optimize the tail
    if n_unvisited <= 5:
        nodes = list(unvisited_nodes)
        best_cost = np.inf
        best_first_node = nodes[0]
        
        for perm in permutations(nodes):
            # Cost: current -> perm[0] -> ... -> perm[-1] -> destination
            cost = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i+1]]
            cost += distance_matrix[perm[-1], destination_node]
            
            if cost < best_cost:
                best_cost = cost
                best_first_node = perm[0]
                
        return int(best_first_node)

    # Detect new instance: if unvisited count is high (near start of tour),
    # it's likely a new instance. We track by the size of unvisited_nodes
    # being >= n-1 (full or near-full).
    n_total = distance_matrix.shape[0]
    is_new_instance = (n_unvisited >= n_total - 1)  # First or second step of a new tour
    
    if is_new_instance and _FIT_PARAMS is None:
        _instance_counter += 1
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=3.0)
    elif is_new_instance and _instance_counter > 0:
        # New instance, refit
        _instance_counter += 1
        _FIT_PARAMS = _fit_params(distance_matrix, current_node, time_budget=2.0)
    
    if _FIT_PARAMS is None:
        # Fallback default
        alpha, gamma, delta = 0.05, 0.0, 0.0
    else:
        alpha, gamma, delta = _FIT_PARAMS
    
    # If deadline is close, use pure nearest neighbor
    if _is_deadline_passed():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    # Compute scores for candidates (vectorized)
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    scores = dists_from_current.copy()
    
    if alpha > 0:
        dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
        scores += alpha * dists_to_dest
    
    if gamma > 0 and n_unvisited > 2:
        # Vectorized 2-step lookahead
        sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        np.fill_diagonal(sub_dist, np.inf)
        min_next = np.min(sub_dist, axis=1)
        scores += gamma * min_next
    
    if delta > 0 and n_unvisited > 2:
        # Isolation penalty
        d_dest_all = distance_matrix[unvisited_nodes, destination_node]
        min_d_dest = np.min(d_dest_all)
        scores += delta * (d_dest_all - min_d_dest)
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
