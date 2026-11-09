import numpy as np

def priority(adjacency):
    # Calculate degrees
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Calculate sum of degrees of neighbors
    # adjacency is bool, degrees is float
    neighbor_degree_sums = adjacency @ degrees
    
    # Score: Primary objective is low degree (min-degree heuristic)
    # Secondary objective is high neighbor degree sum (breaks ties in favor of 
    # vertices adjacent to hubs, which is strategically better for MIS)
    # Using a small epsilon to ensure degree dominates
    epsilon = 0.2
    scores = -degrees + epsilon * neighbor_degree_sums
    
    return scores
