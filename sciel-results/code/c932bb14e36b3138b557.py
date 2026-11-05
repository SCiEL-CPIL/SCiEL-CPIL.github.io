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
    
    We combine eigenvector centrality with degree-based normalization to better
    identify peripheral nodes that are less connected to the core.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    # Handle edge case of empty graph
    if n == 0:
        return np.array([])
    
    # Compute degree vector
    degrees = adjacency.sum(axis=1)
    max_degree = degrees.max() if n > 0 else 1
    
    # Power iteration to find principal eigenvector (eigenvector centrality)
    # Initialize with degree-based vector for better convergence
    rng = np.random.RandomState(42)
    if max_degree > 0:
        x = degrees / max_degree + 1e-6
    else:
        x = rng.rand(n)
    x = x / np.linalg.norm(x)
    
    # Iterate to convergence with more iterations for robustness
    for _ in range(500):
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
    
    # Normalize eigenvector centrality to [0, 1] range
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min > 1e-12:
        eig_norm = (x - x_min) / (x_max - x_min)
    else:
        eig_norm = np.zeros(n)
    
    # Normalize degree to [0, 1] range
    if max_degree > 0:
        deg_norm = degrees / max_degree
    else:
        deg_norm = np.zeros(n)
    
    # Combine: prioritize vertices with low eigenvector centrality AND low degree
    # Weight eigenvector centrality more heavily as it captures global structure
    combined = 0.7 * eig_norm + 0.3 * deg_norm
    
    # Lower combined centrality => higher priority
    # So score = -combined (lower centrality => higher score)
    scores = -combined
    
    return scores
# EVOLVE-BLOCK-END
