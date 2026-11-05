"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track state across calls for adaptive weighting
_tsp_state = {
    'start_time': None,
    'call_count': 0,
    'total_nodes': None
}


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Return-aware nearest neighbor with adaptive weighting.
    
    Scores each candidate by:
        alpha * dist(current, candidate) + (1-alpha) * dist(candidate, destination)
    
    where alpha adapts based on tour progress. Early in the tour, we weight
    toward destination to avoid poor closures; later, we weight toward nearest
    neighbor for local efficiency.
    """
    # Initialize state
    if _tsp_state['start_time'] is None:
        _tsp_state['start_time'] = time.time()
    
    _tsp_state['call_count'] += 1
    
    # Safety: if we haven't seen total nodes yet, infer from matrix size
    if _tsp_state['total_nodes'] is None:
        _tsp_state['total_nodes'] = distance_matrix.shape[0]
    
    # Deadline guard: if we're running out of time, fall back to simple NN
    elapsed = time.time() - _tsp_state['start_time']
    if elapsed > 115.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        # Should not happen, but safe fallback
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Compute distances
    dist_to_candidates = distance_matrix[current_node, unvisited_nodes]
    dist_from_candidates_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Adaptive alpha: starts high (favor destination-awareness) and decreases
    # as we visit more nodes, shifting toward pure nearest-neighbor
    visited_count = _tsp_state['call_count']
    total_nodes = _tsp_state['total_nodes']
    
    if total_nodes > 1:
        progress = visited_count / total_nodes
        # Alpha decays from ~0.6 to ~0.1 as tour progresses
        alpha = 0.6 - 0.5 * progress
    else:
        alpha = 0.5
    
    # Combined score: lower is better
    scores = alpha * dist_to_candidates + (1 - alpha) * dist_from_candidates_to_dest
    
    # Select the candidate with minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
