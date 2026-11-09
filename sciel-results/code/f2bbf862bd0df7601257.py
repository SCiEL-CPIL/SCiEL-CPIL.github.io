import numpy as np

def priority(adjacency):
    # Convert to float for matrix multiplication
    A = adjacency.astype(float)
    
    # Calculate 2-hop connectivity
    # A^2 counts walks of length 2. 
    # (A + A^2) indicates if a node is within distance 1 or 2.
    # We want the size of the 2-neighborhood (including self and 1-hop).
    # Note: A has 0 on diagonal (simple graph). A^2 has deg(i) on diagonal.
    # So (A + A^2) has deg(i) on diagonal.
    # Sum of row i in (A + A^2) = deg(i) + (number of nodes at distance exactly 2).
    # This is a good proxy for "how much of the graph this vertex covers".
    
    A2 = A @ A
    footprint = (A + A2).sum(axis=1)
    
    # Score: Lower footprint is better
    scores = -footprint
    
    return scores
