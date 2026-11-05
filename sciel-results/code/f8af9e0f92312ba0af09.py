"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Convert to numpy if needed
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Degree of each vertex
    degrees = adj.sum(axis=1)
    
    # Sum of degrees of neighbors
    neighbor_degree_sums = adj @ degrees
    
    # Mean degree of neighbors
    # Handle vertices with degree 0
    mean_neighbor_degree = np.zeros(n)
    mask = degrees > 0
    mean_neighbor_degree[mask] = neighbor_degree_sums[mask] / degrees[mask]
    
    # Heuristic: 
    # 1. Prefer vertices with LOW degree (they block fewer vertices).
    # 2. Among those, prefer vertices whose neighbors have HIGH degree.
    #    Why? Picking a low-degree vertex adjacent to high-degree vertices 
    #    "kills" the high-degree vertices (preventing them from being picked 
    #    and blocking many other vertices). This preserves the rest of the 
    #    graph for more independent set members.
    
    score = -degrees + mean_neighbor_degree
    
    return score
# EVOLVE-BLOCK-END
