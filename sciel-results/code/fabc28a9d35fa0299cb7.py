"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Compute degree of each vertex
    degrees = adjacency.sum(axis=1)
    
    # Compute the sum of degrees of neighbors for each vertex
    # This is the second-order degree (sum of neighbor degrees)
    neighbor_degree_sums = adjacency @ degrees
    
    # Reward vertices that are "sheltered": low degree but high neighbor degree sum
    # This prioritizes vertices that can disable high-degree hubs without much cost
    # Use a combination: low own degree is good, high neighbor degree sum is good
    scores = neighbor_degree_sums.astype(float) - degrees.astype(float)
    
    return scores
# EVOLVE-BLOCK-END
