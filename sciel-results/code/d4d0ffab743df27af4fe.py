"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIME_BUDGET = 140.0  # seconds, safety margin before 150s kill

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _time_remaining():
    return _TIME_BUDGET - (time.time() - _get_start_time())


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    if _time_remaining() < 0.5:
        # Emergency: just nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Get distances from current node to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]  # shape (n_unvisited,)
    # Get distances from destination to all unvisited
    d_dest = distance_matrix[destination_node, unvisited_nodes]  # shape (n_unvisited,)
    
    # d_matrix among unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row i, find the two smallest values (excluding self-distance = 0)
    # Use argpartition for efficiency
    if n_unvisited > 2:
        # Get indices of two smallest in each row
        part_idx = np.argpartition(d_unvisited, 1, axis=1)[:, :2]
        # Get the two smallest values
        part_vals = np.take_along_axis(d_unvisited, part_idx, axis=1)
        
        # best_next_cost[i] = min over j != i of d(i,j)
        # d(i,i) = 0 is always the smallest, so the second smallest is our answer
        best_next_cost = part_vals[:, 1]
        best_next_idx = part_idx[:, 1]
    else:
        # n_unvisited == 2
        best_next_cost = np.array([d_unvisited[0, 1], d_unvisited[1, 0]])
        best_next_idx = np.array([1, 0])
    
    # Soft-min surrogate: instead of hard min over next-step, use log-sum-exp
    # This smooths the nonsmooth objective and provides gradient-like signal
    # soft_min(d_unvisited[i, :]) over j != i, with temperature tau
    # We exclude self by setting diagonal to inf before soft-min
    tau = 0.15  # temperature for soft-min
    d_unvisited_no_self = d_unvisited.copy()
    np.fill_diagonal(d_unvisited_no_self, np.inf)
    
    # soft_min: -tau * log(sum(exp(-d/tau)))
    # Numerically stable: subtract max first
    neg_d_over_tau = -d_unvisited_no_self / tau
    max_neg = np.max(neg_d_over_tau, axis=1, keepdims=True)
    # Handle case where all are inf (shouldn't happen for n_unvisited >= 2)
    max_neg = np.where(np.isinf(max_neg), 0, max_neg)
    shifted = neg_d_over_tau - max_neg
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1)) + max_neg.flatten()
    soft_min_cost = -tau * log_sum_exp
    
    # Minimax bottleneck term: max distance from candidate i to any other unvisited
    # This discourages selecting nodes far from the remaining cluster
    if n_unvisited > 1:
        bottleneck = np.max(d_unvisited_no_self, axis=1)
    else:
        bottleneck = np.zeros(n_unvisited)
    
    # Adaptive weights based on remaining nodes
    progress = 1.0 - n_unvisited / 50.0  # rough progress estimate
    
    alpha = 0.5 * (1.0 - progress)  # soft lookahead weight decreases as we finish
    beta = 0.3 * (1.0 - progress)   # destination bias decreases as we finish
    gamma = 0.05 * (1.0 - progress) # minimax bottleneck weight decreases as we finish
    
    # Compute score for each candidate
    # score[i] = d(current, i) + alpha * soft_min_cost[i] + beta * d_dest[i] + gamma * bottleneck[i]
    scores = d_current + alpha * soft_min_cost + beta * d_dest + gamma * bottleneck
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
