"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline guard
_START_TIME = time.time()
# Allow 10 seconds margin before the 150s limit
_DEADLINE = _START_TIME + 140.0


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead heuristic.
    
    Args:
        current_node: int, the current node in the tour.
        destination_node: int, the destination node (usually start).
        unvisited_nodes: np.ndarray, array of unvisited node indices.
        distance_matrix: np.ndarray, NxN matrix of distances.
        
    Returns:
        int, the index of the next node to visit.
    """
    # Handle edge case: no unvisited nodes
    if len(unvisited_nodes) == 0:
        return -1

    # Deadline guard: If we are running out of time, fall back to fast Nearest Neighbor
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    K = len(unvisited_nodes)
    
    # 1. Calculate distance from current node to all unvisited nodes
    # d_cur[i] = distance(current_node, unvisited_nodes[i])
    d_cur = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Calculate the minimum distance from each unvisited node to any other unvisited node
    # This approximates the cost of the *next* edge after visiting unvisited_nodes[i]
    
    if K > 1:
        # Extract the submatrix of distances between unvisited nodes
        # D_sub[i, j] = distance(unvisited_nodes[i], unvisited_nodes[j])
        D_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        
        # Set diagonal to infinity to exclude self-distance
        np.fill_diagonal(D_sub, np.inf)
        
        # Find the minimum distance to any other unvisited node for each node
        # d_next[i] = min_j (distance(unvisited_nodes[i], unvisited_nodes[j])) for j != i
        d_next = np.min(D_sub, axis=1)
    else:
        # If only one node is left, the "next" cost is effectively 0 (or distance to destination, 
        # but since we must pick it, the relative score doesn't matter for selection among candidates)
        d_next = np.array([0.0])

    # 3. Combine scores: Cost to reach candidate + Cost of the next best move from candidate
    scores = d_cur + d_next
    
    # 4. Select the candidate with the minimum total score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
