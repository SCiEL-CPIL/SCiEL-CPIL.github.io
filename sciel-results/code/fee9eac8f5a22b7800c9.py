"""Improved CVRP heuristic combining savings, regret, and 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 148.0  # Safety margin before 150s limit

def _time_ok():
    return (time.time() - _T0) < _DEADLINE

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If time is running out, return depot to close route safely
    if not _time_ok():
        return int(depot)
    
    if len(unvisited_nodes) == 0:
        return int(depot)
    
    # Convert to numpy arrays for efficiency
    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    
    # Filter feasible nodes based on capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)
    
    feas_nodes = unvisited_arr[feas_mask]
    n_nodes = len(feas_nodes)
    
    if n_nodes == 1:
        return int(feas_nodes[0])
    
    # Calculate distances
    d_cur_dep = distance_matrix[current_node, depot]
    d_dep_feas = distance_matrix[depot, feas_nodes]
    d_cur_feas = distance_matrix[current_node, feas_nodes]
    
    # Dynamic Regret-Minimization Framework:
    # The core idea: score each candidate by how much its edge cost deviates
    # from the average edge cost of the remaining feasible set.
    # We prefer nodes whose edge cost is BELOW the average (shorter-than-typical edges),
    # which maintains uniform segment lengths and prevents premature capacity
    # exhaustion on distant nodes.
    
    # Primary signal: regret = d(current, c) - mean(d(current, feasible_set))
    # Negative regret means this edge is shorter than average → prefer it.
    mean_d_cur = np.mean(d_cur_feas)
    regret = d_cur_feas - mean_d_cur  # negative = good (below-average edge)
    
    # Secondary signal: savings-based component to maintain cluster compactness
    # Savings = d(current, depot) + d(depot, c) - d(current, c)
    # High savings means merging this node saves more distance overall.
    savings = d_cur_dep + d_dep_feas - d_cur_feas
    
    # Tertiary signal: soft proximity to other feasible nodes (avoid dead-ends)
    # For each candidate c, compute the distance to its nearest other feasible node.
    # This is a cheap proxy for "can I continue the route efficiently after c?"
    if n_nodes > 1:
        d_feas_feas = distance_matrix[np.ix_(feas_nodes, feas_nodes)]
        np.fill_diagonal(d_feas_feas, np.inf)
        min_neighbor_dist = np.min(d_feas_feas, axis=1)
    else:
        min_neighbor_dist = np.array([d_dep_feas[0]])
    
    # Combine signals with a temperature-scaled soft-max style weighting:
    # score = -alpha * regret + beta * normalized_savings - gamma * normalized_neighbor_dist
    #
    # The regret term is the primary driver (structural change from weighted sum).
    # Savings provides secondary guidance for cluster compactness.
    # Neighbor distance penalizes isolated nodes that would create dead-ends.
    
    # Normalize savings to [0, 1]
    s_min, s_max = np.min(savings), np.max(savings)
    savings_norm = (savings - s_min) / (s_max - s_min + 1e-10)
    
    # Normalize neighbor distance to [0, 1]
    nd_min, nd_max = np.min(min_neighbor_dist), np.max(min_neighbor_dist)
    nd_norm = (min_neighbor_dist - nd_min) / (nd_max - nd_min + 1e-10)
    
    # Regret is already centered (mean ~ 0), so scale it directly.
    # Use a temperature to control sensitivity to outlier edges.
    temperature = 1.0  # Can be tuned; lower = more aggressive avoidance of long edges
    
    # Score: minimize regret (prefer short edges), maximize savings, minimize isolation
    alpha = 1.5 * temperature   # Primary: regret minimization
    beta = 0.8                  # Secondary: savings
    gamma = 0.3                 # Tertiary: avoid dead-ends
    
    scores = -alpha * regret + beta * savings_norm - gamma * nd_norm
    
    # Select the node with the highest score
    best_idx = np.argmax(scores)
    best_node = int(feas_nodes[best_idx])
    
    return best_node
# EVOLVE-BLOCK-END
