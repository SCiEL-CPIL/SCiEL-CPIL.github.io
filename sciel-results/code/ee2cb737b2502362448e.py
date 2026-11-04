import numpy as np

def priority(adjacency):
    # adjacency is a bool matrix n x n
    # Convert to int for matrix multiplication
    A = adjacency.astype(int)
    
    # Calculate degrees
    degrees = A.sum(axis=1).astype(float)
    
    # Calculate sum of neighbor degrees
    # sum_deg_neighbors[i] = sum(degrees[j] for j in neighbors of i)
    # This is A @ degrees
    sum_neighbor_degrees = A @ degrees
    
    # Score: High neighbor degree sum, low own degree
    # We want to pick vertices that are "leaves" attached to "hubs"
    # to remove hubs without removing the leaves.
    # score = sum_neighbor_degrees - degrees
    # To break ties in favor of lower degree vertices (safer), 
    # we can subtract a small multiple of degrees.
    # However, sum_neighbor_degrees is usually much larger.
    # Let's use:
    scores = sum_neighbor_degrees - degrees
    
    # Ensure we return a list or array of floats
    return scores
