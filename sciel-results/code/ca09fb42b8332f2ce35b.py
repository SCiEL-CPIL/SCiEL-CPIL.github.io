"""Program using spectral centrality for Maximum Independent Set priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign scores to vertices based on eigenvector centrality.
    Vertices with lower eigenvector centrality (more peripheral) get higher priority.
    We use power iteration to compute the principal eigenvector of the adjacency matrix.
    
    The idea: vertices that are less central in the graph (lower eigenvector centrality)
    are less entangled with the core structure, so selecting them first leaves more
    non-neighbors available for subsequent selections, potentially yielding a larger
    independent set.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    # Handle edge case of empty graph
    if n == 0:
        return np.array([])
    
    # Compute degree vector
    degrees = adjacency.sum(axis=1)
    
    # Power iteration to find principal eigenvector (eigenvector centrality)
    # Initialize with random positive vector
    rng = np.random.RandomState(42)
    x = rng.rand(n)
    x = x / np.linalg.norm(x)
    
    # Iterate to convergence
    for _ in range(300):
        x_new = adjacency @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-12:
            # Graph has no edges or is disconnected in a way that gives zero vector
            x_new = np.ones(n) / np.sqrt(n)
            break
        x_new = x_new / norm
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-12:
            x = x_new
            break
        x = x_new
    
    # Ensure non-negative (eigenvector centrality should be non-negative for connected graphs)
    x = np.abs(x)
    
    # Combine eigenvector centrality with degree information
    # Use a weighted combination: lower centrality and lower degree => higher priority
    # Normalize eigenvector centrality to [0, 1]
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min > 1e-12:
        x_norm = (x - x_min) / (x_max - x_min)
    else:
        x_norm = np.zeros(n)
    
    # Normalize degree to [0, 1]
    deg_min = degrees.min()
    deg_max = degrees.max()
    if deg_max - deg_min > 1e-12:
        deg_norm = (degrees - deg_min) / (deg_max - deg_min)
    else:
        deg_norm = np.zeros(n)
    
    # Weighted combination: prioritize low centrality and low degree
    # The weight can be tuned; here we use 0.7 for centrality and 0.3 for degree
    combined = 0.7 * x_norm + 0.3 * deg_norm
    
    # Lower combined score => higher priority
    # So score = -combined (lower combined => higher score)
    scores = -combined
    
    return scores
# EVOLVE-BLOCK-END
