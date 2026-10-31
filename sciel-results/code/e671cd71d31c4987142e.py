"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None
_START_TIME = None

def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        # 150s wall-clock limit; use 145s as safety margin
        _DEADLINE = _START_TIME + 145.0

def _time_remaining():
    global _DEADLINE
    if _DEADLINE is None:
        return 145.0
    return max(0.0, _DEADLINE - time.time())

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node using a look-ahead heuristic that balances
    immediate distance with future connectivity cost.
    
    Uses a 2-step look-ahead: for each candidate next node, estimate
    the cost of going there plus the cost of the best subsequent move
    (or return to destination if only one node remains).
    """
    _ensure_deadline()
    
    if len(unvisited_nodes) == 0:
        # Should not happen, but handle gracefully
        return int(destination_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    if len(unvisited_nodes) == 2:
        # Choose the one that minimizes: dist(current, A) + dist(A, B) + dist(B, dest)
        # vs dist(current, B) + dist(B, A) + dist(A, dest)
        a, b = unvisited_nodes
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b] + distance_matrix[b, destination_node]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a] + distance_matrix[a, destination_node]
        return int(a if cost_a <= cost_b else b)
    
    # For larger sets, use a weighted scoring approach
    # Score = alpha * dist(current, candidate) + beta * min_dist(candidate, remaining_unvisited)
    # This encourages choosing nodes that are close now AND well-connected to others
    
    # Fast path: if time is running out, fall back to nearest neighbor
    if _time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    # Compute immediate distances
    immediate = distance_matrix[current_node, unvisited_nodes]  # shape (n_unvisited,)
    
    # For each candidate, compute the minimum distance to other unvisited nodes
    # (excluding itself) to estimate future connectivity
    # This is an O(n^2) operation but n is at most ~50
    
    # dist_matrix for unvisited nodes among themselves
    sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # shape (n_unvisited, n_unvisited)
    
    # For each candidate i, min distance to any other unvisited node j != i
    # Set diagonal to inf to exclude self
    np.fill_diagonal(sub_dm, np.inf)
    min_future = np.min(sub_dm, axis=1)  # shape (n_unvisited,)
    
    # Also consider the distance from candidate to destination (important for later steps)
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]  # shape (n_unvisited,)
    
    # Weighted score: lower is better
    # alpha=1.0 for immediate, beta=0.5 for future connectivity, gamma=0.3 for destination proximity
    # The gamma term helps avoid getting stuck far from destination late in the tour
    alpha = 1.0
    beta = 0.5
    gamma = 0.3
    
    # Normalize each component to [0,1] range for fair weighting
    imm_min, imm_max = np.min(immediate), np.max(immediate)
    fut_min, fut_max = np.min(min_future), np.max(min_future)
    dest_min, dest_max = np.min(dist_to_dest), np.max(dist_to_dest)
    
    imm_range = imm_max - imm_min if imm_max > imm_min else 1.0
    fut_range = fut_max - fut_min if fut_max > fut_min else 1.0
    dest_range = dest_max - dest_min if dest_max > dest_min else 1.0
    
    score = (alpha * (immediate - imm_min) / imm_range +
             beta * (min_future - fut_min) / fut_range +
             gamma * (dist_to_dest - dest_min) / dest_range)
    
    return int(unvisited_nodes[np.argmin(score)])

# EVOLVE-BLOCK-END
