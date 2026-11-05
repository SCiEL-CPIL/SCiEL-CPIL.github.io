"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """Score vertices using Fiedler vector (spectral cut) + degree.
    
    The Fiedler vector identifies vertices minimally entangled in connectivity.
    Vertices with large |Fiedler| values are on the 'periphery' relative to
    the graph's main cut, making them good candidates for independent set.
    We combine this with negative degree (prefer low-degree vertices).
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Compute degree
    degree = A.sum(axis=1)
    
    # Compute Laplacian L = D - A
    D = np.diag(degree)
    L = D - A
    
    # Compute eigenvectors of Laplacian
    # For n=200, full eigendecomposition is fast
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
    except Exception:
        # Fallback to degree-based scoring
        return -degree.astype(float)
    
    elapsed = time.time() - start_time
    if elapsed > deadline * 0.8:
        # Safety margin: return degree-based score
        return -degree.astype(float)
    
    # Fiedler vector is the eigenvector corresponding to the second-smallest eigenvalue
    # eigenvalues are sorted in ascending order by eigh
    fiedler = eigenvectors[:, 1]
    
    # Score: prioritize vertices with large |Fiedler| (peripheral) AND low degree
    # Normalize Fiedler values to [0, 1] range for combination
    fiedler_abs = np.abs(fiedler)
    fiedler_min = fiedler_abs.min()
    fiedler_max = fiedler_abs.max()
    if fiedler_max > fiedler_min:
        fiedler_norm = (fiedler_abs - fiedler_min) / (fiedler_max - fiedler_min)
    else:
        fiedler_norm = np.ones(n) / n
    
    # Normalize degree to [0, 1]
    deg_min = degree.min()
    deg_max = degree.max()
    if deg_max > deg_min:
        deg_norm = (degree - deg_min) / (deg_max - deg_min)
    else:
        deg_norm = np.zeros(n)
    
    # Combined score: high Fiedler peripherality + low degree
    # Higher score = better candidate
    score = fiedler_norm - 0.5 * deg_norm
    
    elapsed = time.time() - start_time
    if elapsed > deadline * 0.9:
        # Very close to deadline, return immediately
        return score.astype(float)
    
    return score.astype(float)
# EVOLVE-BLOCK-END
