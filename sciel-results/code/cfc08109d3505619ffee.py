"""
2-Step Lookahead TSP Constructive Heuristic.

Strategy:
Instead of simply picking the nearest neighbor (or a phase-blended neighbor),
this heuristic looks two steps ahead. For each candidate next node `c`, it
estimates the cost of moving from `current` to `c` and then from `c` to its
best possible successor `s` (among remaining unvisited nodes).

Score for candidate c = dist(current, c) + min_{s in unvisited\{c}} dist(c, s)

This avoids "dead ends" where a node is close to the current position but
far from all other remaining nodes, which often leads to expensive closing
edges or long detours later in the tour.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management
_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        return 0.0
    return time.time() - _start_time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    
    Args:
        current_node (int): The node currently being visited.
        destination_node (int): The starting node (to be visited last to close tour).
        unvisited_nodes (np.ndarray): Array of node indices not yet visited.
        distance_matrix (np.ndarray): NxN matrix of Euclidean distances.
    
    Returns:
        int: The index of the next node to visit.
    """
    n_unvisited = len(unvisited_nodes)
    
    # Base cases
    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check time budget - if running low, fall back to fast greedy
    elapsed = _get_elapsed()
    if elapsed > _TIMEOUT:
        # Fallback: Pure Nearest Neighbor (fastest)
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Standard case: 2-step lookahead
    # 1. Get distance from current to all candidates
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. For each candidate, find the minimum distance to any other unvisited node (exit cost)
    #    We compute the min distance from each candidate to the set of *other* unvisited nodes.
    #    Note: We don't need to exclude the candidate itself if we handle the min carefully,
    #    but strictly speaking, the next-next node must be different from the candidate.
    
    # Efficient vectorization:
    # Get the submatrix of distances between all unvisited nodes
    # unvisited_dist_matrix[i, j] is distance from unvisited_nodes[i] to unvisited_nodes[j]
    unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row i (candidate i), we want the min distance to any j != i.
    # We can set the diagonal to infinity before taking the min along axis 1.
    # This gives the "best exit" cost for each candidate.
    
    # Create a copy to avoid modifying the original view if possible, or use a mask
    # Since n_unvisited is small (<=50), copying is cheap.
    exit_costs = np.min(np.where(np.eye(n_unvisited, dtype=bool), np.inf, unvisited_dist_matrix), axis=1)
    
    # 3. Combine costs: Entry cost + Exit cost
    #    score_i = dist(current, candidate_i) + exit_cost_i
    scores = dist_from_current + exit_costs
    
    # 4. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
