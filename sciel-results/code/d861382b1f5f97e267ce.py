"""Item-pair attractiveness heuristic for ACO bin packing."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Compute an item-by-item attractiveness matrix for placing item j right after
    item i in the same bin. Larger values mean more attractive.
    
    Strategy: Combine tight-fit preference (inverse waste) with a spectral
    compatibility coordinate derived from the graph Laplacian of the item
    compatibility graph. The spectral coordinate captures global structure
    of which items naturally cluster together, while the inverse waste
    captures local tight-fit quality.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    d = np.asarray(demands, dtype=float)
    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    
    cap = float(capacity)
    pair = d[:, None] + d[None, :]
    fits = pair <= cap
    remaining = cap - pair  # space left if i and j share a bin
    
    # Base heuristic: inverse of remaining space (tight fits are more attractive)
    # Use 1/(1 + remaining) to avoid division by zero and keep values bounded
    base_h = np.full((n, n), 1e-9, dtype=np.float64)
    base_h[fits] = 1.0 / (1.0 + remaining[fits])
    
    # Check time budget
    if time.time() > deadline - 5.0:
        # Not enough time for spectral computation, return base heuristic
        base_h[~np.isfinite(base_h)] = 1e-9
        base_h[base_h <= 0] = 1e-9
        return base_h
    
    # Build compatibility graph for spectral analysis
    # Edge weight: how well two items fit together (inverse waste, only for fitting pairs)
    W = np.zeros((n, n), dtype=np.float64)
    W[fits] = 1.0 / (1.0 + remaining[fits])
    
    # Compute degree matrix
    degree = np.sum(W, axis=1)
    
    # Avoid division by zero for isolated nodes
    degree_safe = np.where(degree > 0, degree, 1.0)
    
    # Construct normalized Laplacian: L = I - D^(-1/2) W D^(-1/2)
    D_inv_sqrt = 1.0 / np.sqrt(degree_safe)
    L = np.eye(n) - (D_inv_sqrt[:, None] * W) * D_inv_sqrt[None, :]
    
    # Ensure symmetry (numerical stability)
    L = (L + L.T) / 2.0
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # If eigendecomposition fails, return base heuristic
        base_h[~np.isfinite(base_h)] = 1e-9
        base_h[base_h <= 0] = 1e-9
        return base_h
    
    # Check time budget
    if time.time() > deadline - 5.0:
        base_h[~np.isfinite(base_h)] = 1e-9
        base_h[base_h <= 0] = 1e-9
        return base_h
    
    # The Fiedler vector is the eigenvector corresponding to the second smallest
    # eigenvalue (index 1, since eigenvalues are sorted in ascending order)
    fiedler = eigenvectors[:, 1]
    
    # Make coordinates positive by shifting
    # The Fiedler vector can have negative values; we want positive coordinates
    # Shift so that all values are positive
    fiedler_shifted = fiedler - np.min(fiedler) + 0.1  # add small offset to avoid zeros
    
    # Create compatibility coordinate matrix: product of coordinates
    # This captures global clustering structure
    coord_matrix = fiedler_shifted[:, None] * fiedler_shifted[None, :]
    
    # Normalize coord_matrix to have similar scale as base_h
    coord_max = np.max(coord_matrix)
    if coord_max > 0:
        coord_matrix = coord_matrix / coord_max
    
    # Combine base heuristic with spectral coordinate
    # Use a weighted combination: 0.7 * base + 0.3 * spectral
    # The spectral component adds global structure awareness
    combined_h = 0.7 * base_h + 0.3 * coord_matrix
    
    # Apply fitting mask: non-fitting pairs get 1e-9
    combined_h[~fits] = 1e-9
    
    # Ensure all values are positive and finite
    combined_h[~np.isfinite(combined_h)] = 1e-9
    combined_h[combined_h <= 0] = 1e-9
    
    return combined_h
# EVOLVE-BLOCK-END
