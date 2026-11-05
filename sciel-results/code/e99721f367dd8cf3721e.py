"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    """
    Computes a priority score for each vertex to maximize the size of the
    independent set selected by a greedy decoder.
    
    Heuristic:
    Prioritize vertices with low degree, especially those that are neighbors
    of high-degree vertices (hubs). Selecting such 'leaf' vertices blocks the
    hubs but allows other leaves to be selected. Selecting hubs blocks many
    leaves.
    
    Score = mean_degree_of_neighbors - own_degree
    """
    # Deadline guard (not strictly necessary for N=200, but good practice)
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Convert to float for matrix operations if not already
    # adjacency is a bool matrix
    deg = adjacency.sum(axis=1).astype(float)
    
    # Calculate sum of degrees of neighbors for each vertex
    # This is equivalent to adjacency @ deg
    sum_neighbor_deg = adjacency @ deg
    
    # Calculate mean degree of neighbors
    # Handle division by zero for isolated nodes
    avg_neighbor_deg = np.zeros_like(deg)
    mask = deg > 0
    avg_neighbor_deg[mask] = sum_neighbor_deg[mask] / deg[mask]
    
    # The heuristic score
    # High score = High priority
    # We want to pick low degree nodes (so -deg)
    # We want to pick nodes with high neighbor degrees (so +avg_neighbor_deg)
    scores = avg_neighbor_deg - deg
    
    # Optional: Add a tiny epsilon based on index to break ties deterministically
    # but favoring lower indices? Or just leave it.
    # The prompt says "fixed greedy decoder". If it uses np.argsort, ties are broken
    # by stable sort (original order).
    
    if time.time() > deadline:
        # Fallback to simple degree-based score if computation was too slow (unlikely)
        return -deg
        
    return scores
# EVOLVE-BLOCK-END
