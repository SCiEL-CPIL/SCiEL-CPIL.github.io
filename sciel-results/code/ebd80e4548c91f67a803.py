import numpy as np
import time

# Deadline guard: track elapsed time to avoid timeout
_START_TIME = [None]
_DEADLINE = 130.0  # Safety margin before 150s limit

def _check_deadline():
    """Return True if we should stop computing and return immediately."""
    if _START_TIME[0] is None:
        return False
    return (time.time() - _START_TIME[0]) > _DEADLINE

def _softmin(values, temperature=1.0):
    """Soft minimum: differentiable approximation of min that considers multiple small values."""
    if len(values) == 0:
        return np.inf
    if len(values) == 1:
        return values[0]
    # Normalize by temperature for numerical stability
    v = np.asarray(values, dtype=np.float64)
    min_v = np.min(v)
    exps = np.exp(-(v - min_v) / temperature)
    if exps.sum() == 0:
        return min_v
    return min_v - temperature * np.log(exps.sum())

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a softmin-based multi-step lookahead with MST-aware penalty.
    """
    if _START_TIME[0] is None:
        _START_TIME[0] = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return destination_node 
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        u1 = int(unvisited_nodes[0])
        u2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, u1] + distance_matrix[u1, u2] + distance_matrix[u2, destination_node]
        cost2 = distance_matrix[current_node, u2] + distance_matrix[u2, u1] + distance_matrix[u1, destination_node]
        return u1 if cost1 <= cost2 else u2

    # Deadline check: if we're running low on time, fall back to greedy nearest neighbor
    if _check_deadline():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])

    # General case: multi-step lookahead with softmin
    # For each candidate u, compute:
    #   score = d(current, u) + softmin_{v in unvisited\{u}} [d(u,v) + alpha * d(v, destination)]
    # This balances immediate cost with future connectivity to destination.
    
    # Get distances from current to all unvisited
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all unvisited
    dist_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Temperature for softmin: lower = more like min, higher = more balanced
    # Use a moderate temperature that scales with problem size
    temp = 0.5 * np.sqrt(n_unvisited)
    
    best_score = np.inf
    best_node = int(unvisited_nodes[0])
    
    # Precompute: for each candidate u, we need softmin over v != u of [d(u,v) + alpha*d(v,dest)]
    # Vectorize: for each u, compute the vector of d(u, v) + alpha * d(v, dest) for all v, mask out v=u, then softmin
    
    # alpha: weight on future distance to destination
    alpha = 0.3
    
    # Build the full submatrix of distances among unvisited nodes
    # dist_sub[i,j] = distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
    dist_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each candidate u (row i), the future cost vector is dist_sub[i, :] + alpha * dist_to_dest
    # Then mask out column i (v = u) and take softmin
    future_costs = dist_sub + alpha * dist_to_dest[None, :]  # shape (n, n)
    
    # Mask diagonal: for row i, set column i to inf
    mask = np.eye(n_unvisited, dtype=bool)
    future_costs_masked = future_costs.copy()
    future_costs_masked[mask] = np.inf
    
    # Compute softmin for each row
    # softmin_i = min_i - temp * log(sum(exp(-(future_costs_masked[i,:] - min_i) / temp)))
    min_vals = np.min(future_costs_masked, axis=1)
    exps = np.exp(-(future_costs_masked - min_vals[:, None]) / temp)
    # Handle numerical issues where exps might underflow to 0
    exps_sum = np.sum(exps, axis=1)
    exps_sum = np.maximum(exps_sum, 1e-300)
    softmin_vals = min_vals - temp * np.log(exps_sum)
    
    # Total score for each candidate u
    scores = dist_from_current + softmin_vals
    
    # Add MST-aware penalty: penalize candidates that would create a "bridge" edge
    # A bridge edge is one where removing it would disconnect the remaining graph significantly.
    # Approximation: if u's nearest neighbor in unvisited\{u} is far, it might be a bridge.
    # Use the second-nearest neighbor distance as a proxy for connectivity risk.
    
    # For each u, find the two smallest distances to other unvisited nodes
    # We already have future_costs_masked; the min is the nearest, second min is next
    sorted_futures = np.sort(future_costs_masked, axis=1)
    # sorted_futures[:, 0] = nearest, sorted_futures[:, 1] = second nearest
    # If there's only 1 other node, second nearest is inf
    if n_unvisited > 2:
        nearest_d = sorted_futures[:, 0]
        second_nearest_d = sorted_futures[:, 1]
        # Bridge penalty: if second nearest is much larger than nearest, u is somewhat isolated
        # Penalty = beta * max(0, second_nearest_d - nearest_d)
        bridge_penalty = 0.2 * np.maximum(0, second_nearest_d - nearest_d)
    else:
        bridge_penalty = np.zeros(n_unvisited)
    
    scores += bridge_penalty
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
