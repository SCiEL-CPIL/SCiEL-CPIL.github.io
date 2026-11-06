"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state to track time and cache computations
_start_time = None
_time_limit = 120.0  # seconds
_safety_margin = 5.0  # seconds
_cache = {}


def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _time_exceeded():
    return _get_elapsed() > (_time_limit - _safety_margin)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check time budget
    if _time_exceeded():
        # Fallback to simple nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Use penalty-aware lookahead: score = immediate_edge + alpha * min_next_hop + beta * destination_closure
    # This discourages choices that strand the tour far from the return city
    
    if len(unvisited_nodes) <= 2:
        # For small sets, use nearest neighbor to avoid overhead
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Compute scores for each candidate
    # Immediate edge cost
    immediate = distance_matrix[current_node, unvisited_nodes]
    
    # Min next-hop: for each candidate, the minimum distance to any other unvisited node
    # This is a proxy for how "stranded" the candidate would be
    # We can approximate this by looking at the minimum distance from each candidate to others
    # But this is O(n^2) which might be too slow. Let's use a simpler proxy.
    
    # Simpler proxy: distance to destination (encourages moving toward the return city)
    dest_dist = distance_matrix[unvisited_nodes, destination_node]
    
    # Combine: score = immediate + 0.5 * dest_dist
    # The weight 0.5 balances immediate cost with future closure
    scores = immediate + 0.5 * dest_dist
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
