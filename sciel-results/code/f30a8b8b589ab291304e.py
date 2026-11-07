import numpy as np

def heuristics(distance_matrix):
    # Increasing the exponent to 3.0 makes the heuristic even more sensitive to short edges.
    # This stronger focus on local optimality can help ACO find better tours by
    # reducing the probability of selecting longer, less promising edges.
    
    eps = 1e-9
    return (distance_matrix + eps) ** -3.0
