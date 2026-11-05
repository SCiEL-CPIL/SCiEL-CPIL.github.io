"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Score vertices for greedy independent set selection.
    Returns a score per vertex; higher score = selected earlier.
    Uses a 2-coloring heuristic derived from the principal eigenvector of the adjacency matrix.
    Vertices with larger magnitude in the dominant color class are preferred.
    """
    n = adjacency.shape[0]
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    # Convert to float for computations
    A = adjacency.astype(float)

    # Degree
    degree = A.sum(axis=1)

    # Power iteration for principal eigenvector
    max_iter = 500
    v = np.ones(n) / np.sqrt(n)
    v = v / (np.linalg.norm(v) + 1e-12)

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new = v_new / norm
        if np.linalg.norm(v_new - v) < 1e-10:
            v = v_new
            break
        v = v_new

    eigenvector = v  # principal eigenvector

    # Compute signed components
    # The principal eigenvector of an ER graph with p > 0.5 has both positive and negative entries
    # We use the sign to define a 2-coloring
    sign = np.sign(eigenvector)
    
    # For vertices with zero or near-zero eigenvector value, assign based on degree
    zero_mask = np.abs(eigenvector) < 1e-8
    if np.any(zero_mask):
        # Assign these vertices to the color class with fewer members to balance
        pos_count = np.sum(sign > 0)
        neg_count = np.sum(sign < 0)
        if pos_count <= neg_count:
            sign[zero_mask] = 1
        else:
            sign[zero_mask] = -1

    # Magnitude of eigenvector component
    magnitude = np.abs(eigenvector)

    # For the dominant color class (larger magnitude), we want higher priority
    # But we also want to consider degree: lower degree vertices are easier to include
    # So we combine: prefer vertices in the dominant class with lower degree
    
    # Determine which color class is "dominant" by summing magnitudes
    pos_mag_sum = np.sum(magnitude[sign > 0])
    neg_mag_sum = np.sum(magnitude[sign < 0])
    
    # Normalize features
    def normalize(x):
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-12:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    norm_degree = normalize(degree)
    norm_magnitude = normalize(magnitude)

    # Priority score:
    # - Higher magnitude in the dominant color class = higher priority
    # - Lower degree = higher priority (easier to include in independent set)
    
    # Create a score that favors vertices in the dominant class
    if pos_mag_sum >= neg_mag_sum:
        # Positive class is dominant
        class_score = np.where(sign > 0, norm_magnitude, -norm_magnitude)
    else:
        # Negative class is dominant
        class_score = np.where(sign < 0, norm_magnitude, -norm_magnitude)

    # Combine class score and inverse degree
    # Weight: 70% class score, 30% inverse degree
    w_class = 0.7
    w_degree = 0.3

    priority_scores = w_class * class_score + w_degree * (1.0 - norm_degree)

    # Ensure scores are distinct for stable sorting
    # Add a tiny random perturbation based on vertex index to break ties
    perturbation = np.linspace(0, 1e-6, n)
    priority_scores += perturbation

    return priority_scores
# EVOLVE-BLOCK-END
