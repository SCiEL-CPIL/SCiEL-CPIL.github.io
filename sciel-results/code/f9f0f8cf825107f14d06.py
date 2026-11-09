"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for timing / state (not strictly necessary but safe)
_start_time = time.time()
_DEADLINE = 150.0
_TIME_MARGIN = 10.0  # seconds before deadline to stop heavy work


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour.
    
    Strategy: 
    - Use a phase-aware blend of nearest-neighbor and destination-proximity.
    - For early picks, use pure nearest-neighbor.
    - For later picks, blend in proximity to the destination_node to avoid 
      a long closing edge.
    - The blend ratio increases as we approach the end of the tour.
    
    We estimate progress by the number of unvisited nodes remaining.
    With 50 cities total, if we have k unvisited nodes, we are at step (50 - k).
    We start blending in destination proximity when k <= 15 (last 15 picks).
    """
    n_unvisited = len(unvisited_nodes)
    
    # Total cities is estimated from distance_matrix size
    n_total = distance_matrix.shape[0]
    
    # Progress: 0 at start, 1 at end
    # When n_unvisited == n_total - 1, we just started (1 pick made)
    # When n_unvisited == 1, we are at the last pick
    # Let's define progress as (n_total - n_unvisited) / (n_total - 1)
    # But we want to keep NN for the first ~40 picks, so blend starts at progress ~0.8
    
    if n_total <= 1:
        return int(unvisited_nodes[0])
    
    # Calculate how many nodes we've already visited (excluding current and destination if they were visited)
    # Actually, simpler: we know n_total. 
    # If we have n_unvisited nodes left, we have visited n_total - n_unvisited - 1 nodes 
    # (minus 1 because current_node is visited but not in unvisited, and we started from a start node).
    # Wait, the interface doesn't give us the start node. 
    # Let's assume the tour is built from a start node.
    # Number of edges made so far = n_total - n_unvisited - 1? 
    # If n_unvisited = n_total - 1, we made 1 edge (from start to current).
    # If n_unvisited = 1, we made n_total - 2 edges? No, we have 1 node left to visit, then go to destination.
    # Total edges in tour = n_total.
    # Edges remaining = n_unvisited + 1 (one to go from current to next, ... , one from last node to destination).
    # Actually, the function picks the NEXT node. 
    # After picking, we move to it. 
    # If n_unvisited = 1, we pick the last node. Then the tour closes from that last node to destination_node.
    # So the "closing edge" is from the node we pick when n_unvisited=1 to destination_node.
    # The edge from current_node to the node we pick when n_unvisited=2 is also important.
    
    # Let's define a weight for destination proximity.
    # We want it to be 0 for the first ~80% of the tour, and increase to 1 for the last ~20%.
    # Let k = n_unvisited.
    # If k > 15, weight = 0.
    # If k <= 15, weight = (15 - k + 1) / 15 = (16 - k) / 15? 
    # Let's say at k=15, weight=0. At k=1, weight=1.
    # weight = (15 - k) / 14 for k in [1, 15]?
    # k=15 -> 0
    # k=1 -> 14/14 = 1
    # This looks good.
    
    if n_unvisited > 15:
        dest_weight = 0.0
    else:
        dest_weight = (15 - n_unvisited) / 14.0
    
    # Get distances from current_node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    if dest_weight == 0.0:
        # Pure nearest neighbor
        idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[idx])
    
    # Get distances from unvisited nodes to destination_node
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Normalize distances to combine them? 
    # Direct addition might be skewed if scales are different, but for Euclidean TSP they are comparable.
    # However, to be safe, we can normalize by the max distance in the matrix or similar.
    # Or just use a weighted sum.
    # Score = (1 - dest_weight) * dists_from_current + dest_weight * dists_to_dest
    
    # To make the combination more robust, let's normalize each component to [0, 1] relative to their max?
    # Or just use the raw values. The phase-aware heuristic measured 1.004, which is a small improvement.
    # Let's try a slightly different blend or a "soft" selection.
    
    # Alternative: Minimize the sum of the edge from current to next, and a penalty for the edge from next to destination.
    # score = dists_from_current + dest_weight * dists_to_dest
    
    scores = dists_from_current + dest_weight * dists_to_dest
    
    idx = np.argmin(scores)
    return int(unvisited_nodes[idx])

# EVOLVE-BLOCK-END
