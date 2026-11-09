"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 145.0  # Safety margin before 150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node for the TSP tour using a phase-aware heuristic.
    Uses Nearest Neighbor for the first ~80% of the tour, then blends in
    proximity to the destination node to minimize the closing edge length.
    """
    # Deadline guard: if we are running out of time, fall back to the fastest option
    if time.time() - _start_time > _TIME_LIMIT:
        if len(unvisited_nodes) == 0:
            return 0
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    N = distance_matrix.shape[0]
    M = len(unvisited_nodes)
    
    if M == 0:
        return 0

    # Define the threshold for when to start blending destination proximity.
    # Based on knowledge: "first 40 picks" (out of 50) are pure NN.
    # This corresponds to the last 20% of nodes (10 nodes) for blending.
    threshold = max(1, int(N * 0.2))
    
    # Calculate the weight (alpha) for the destination distance.
    # We want alpha to be 0 when M >= threshold, and grow towards 1 as M decreases.
    # At M=1 (last step), we want alpha to be 1.0 to minimize d(cur, last) + d(last, dest).
    if M >= threshold:
        alpha = 0.0
    else:
        # Linear ramp from 0 to 1
        # At M = threshold - 1, alpha should be small.
        # At M = 1, alpha should be 1.
        # Range of M: [1, threshold - 1]
        # Range of alpha: [1, ~0]
        # Let's map M=1 -> 1.0 and M=threshold -> 0.0
        denom = threshold - 1
        if denom <= 0:
            alpha = 1.0
        else:
            alpha = (threshold - M) / denom
            if alpha > 1.0:
                alpha = 1.0
            if alpha < 0.0:
                alpha = 0.0

    if alpha == 0.0:
        # Pure Nearest Neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    else:
        # Weighted Score: distance to current + alpha * distance to destination
        d_cur = distance_matrix[current_node, unvisited_nodes]
        d_dest = distance_matrix[destination_node, unvisited_nodes]
        scores = d_cur + alpha * d_dest
        return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
