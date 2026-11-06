"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a priority score to each vertex for a greedy Maximum Independent Set decoder.
    
    Strategy: Use a combination of degree and local triangle count.
    - Low degree is better (fewer conflicts).
    - High number of triangles (edges among neighbors) is better, because
      densely-connected neighbors cannot all be in an independent set anyway,
      so picking the vertex is more efficient.
    
    Score = -degree + triangle_count
    """
    start_time = time.time()
    DEADLINE = 120.0  # Safety margin before 150s kill
    
    # Convert to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=bool)
    
    n = adjacency.shape[0]
    
    # Compute degrees
    deg = adjacency.sum(axis=1).astype(float)
    
    # Compute number of triangles containing each vertex
    # triangles[v] = 0.5 * (A^3)[v, v]
    A2 = adjacency @ adjacency
    A3 = A2 @ adjacency
    triangles = 0.5 * np.diag(A3)
    
    # Score: -degree + triangles
    score = -deg + triangles
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Fallback to simple degree-based score if we're running out of time
        return -deg
    
    # Ensure the score is a 1D array of floats
    return score.astype(float)
# EVOLVE-BLOCK-END
