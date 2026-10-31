import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # Safety margin before 150s

def _time_exceeded():
    return time.time() > _DEADLINE

def _nearest_neighbor_fallback(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Fallback: pick nearest neighbor to current."""
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    dists = distance_matrix[current_node, unvisited_nodes]
    return int(unvisited_nodes[np.argmin(dists)])

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Regret-based constructive heuristic for TSP.
    
    For each candidate next node c, compute a 'regret' score that reflects
    how much more expensive it would be to visit c later versus now.
    
    The score combines:
    1. Immediate distance from current to c
    2. A regret term: how much the connection options diminish if we delay c
    3. A destination-aware term to close the tour efficiently
    
    This is a structural departure from simple 2-step lookahead.
    """
    if _time_exceeded():
        return _nearest_neighbor_fallback(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # For the last two nodes, pick the one that minimizes the total path
        # current -> c -> other -> destination
        c1, c2 = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return int(c1 if cost1 <= cost2 else c2)
    
    # Get distances from current_node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from destination to all unvisited nodes
    dists_from_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Submatrix of distances among unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    
    # For regret computation, we need to understand the connectivity landscape.
    # For each candidate c_i, compute:
    # - min_dist_to_others: min distance from c_i to any other unvisited node
    # - second_min_dist_to_others: second min distance (to measure how unique the best connection is)
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(sub_matrix, np.inf)
    
    # Sort each row to get min and second min
    sorted_rows = np.sort(sub_matrix, axis=1)
    min_dist_to_others = sorted_rows[:, 0]
    second_min_dist_to_others = sorted_rows[:, 1]
    
    # Regret-based scoring:
    # The idea: if a node has a very low min_dist_to_others but a high second_min,
    # it's "well-connected" and less urgent. If it has a high min_dist_to_others,
    # it's isolated and should be visited soon.
    
    # Regret score: penalize nodes that are far from current AND have poor connectivity
    # We want to minimize this score.
    
    # Component 1: Immediate cost
    immediate = dists_from_current
    
    # Component 2: Connectivity regret
    # If a node's best connection is much better than its second best, it's "special"
    # and we might want to visit it now to lock in that cheap edge.
    # Conversely, if all connections are similar, we have more flexibility.
    # Use the gap between min and second min as a measure of "specialness"
    connectivity_gap = second_min_dist_to_others - min_dist_to_others
    
    # Nodes with large gaps are more "urgent" to visit (their best edge might be lost)
    # But we also want to visit nodes that are close to current.
    
    # Component 3: Destination awareness
    # Nodes that are close to the destination might be better saved for later
    # (to close the tour cheaply), unless they're also close to current.
    
    # Combined score:
    # score = w1 * immediate + w2 * min_dist_to_others + w3 * destination_penalty
    
    # The destination penalty should be negative for nodes close to destination
    # (we want to save them for the end), but this is tricky in a greedy setting.
    
    # Alternative approach: use regret more directly.
    # For each candidate c, compute what the "cost" would be if we didn't visit it now.
    # This is hard to compute exactly, so we approximate.
    
    # Let's use a weighted combination:
    # score = alpha * dist(current, c) + beta * min_dist_to_others[c] - gamma * dist(dest, c)
    
    # The negative gamma term encourages saving destination-close nodes for later.
    # But we need to be careful: if we save all destination-close nodes, we might
    # end up with a bad tour structure.
    
    # Let's try:
    alpha = 1.0
    beta = 0.5
    gamma = 0.3
    
    score = alpha * immediate + beta * min_dist_to_others - gamma * dists_from_dest
    
    # Additionally, add a regret bonus for nodes with high connectivity gaps
    # (they're "special" and should be visited soon)
    # Normalize the gap to [0, 1] range
    max_gap = np.max(connectivity_gap) if np.max(connectivity_gap) > 0 else 1.0
    normalized_gap = connectivity_gap / max_gap
    
    # Add a small penalty for high-gap nodes (they're more flexible, so less urgent)
    # Wait, actually high gap means the best edge is much better than others,
    # so it's MORE urgent to visit now to lock in that cheap edge.
    # So we should SUBTRACT a term for high gap (making the score lower/more attractive).
    
    delta = 0.2
    score -= delta * normalized_gap
    
    best_idx = np.argmin(score)
    
    return int(unvisited_nodes[best_idx])
