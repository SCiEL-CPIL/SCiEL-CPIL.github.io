import numpy as np

def heuristics(distance_matrix):
    dm = np.asarray(distance_matrix, dtype=float)
    # Avoid division by zero
    d = dm + 1e-9
    # Use inverse square root distance
    # This is less greedy than 1/d, potentially allowing better exploration
    h = 1.0 / np.sqrt(d)
    # Ensure positive
    h = np.maximum(h, 1e-9)
    return h
