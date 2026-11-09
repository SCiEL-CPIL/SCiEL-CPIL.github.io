"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a 2-step lookahead heuristic.
    For each candidate, score = dist(current, candidate) + dist(candidate, destination).
    This avoids myopic choices that lead to expensive final connections.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Score each unvisited node by the sum of:
    # 1. Distance from current node to the candidate
    # 2. Distance from the candidate to the destination node
    # This 2-step lookahead balances immediate cost with future cost.
    dists_to_candidate = distance_matrix[current_node, unvisited_nodes]
    dists_candidate_to_dest = distance_matrix[unvisited_nodes, destination_node]
    scores = dists_to_candidate + dists_candidate_to_dest
    
    # Pick the candidate with the minimum score
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
