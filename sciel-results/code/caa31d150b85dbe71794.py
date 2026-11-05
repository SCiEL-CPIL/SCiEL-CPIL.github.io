"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute vertex scores for greedy Maximum Independent Set selection.
    
    Strategy: Use a combination of spectral (Fiedler vector) information and
    local structural features to identify vertices that are:
    1. On the "periphery" of dense cores (high |Fiedler| value)
    2. Have low degree (less likely to block many other candidates)
    3. Have neighbors with high degree (selecting them removes valuable hubs)
    
    The Fiedler vector identifies vertices that are minimally entangled in the
    graph's connectivity, which correlates with being on the boundary of dense
    clusters - ideal for independent set selection.
    """
    start_time = time.time()
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    # Convert to float for matrix operations
    A = adjacency.astype(float)
    
    # Compute degree vector
    degrees = A.sum(axis=1)
    
    # Compute Laplacian L = D - A
    D = np.diag(degrees)
    L = D - A
    
    # Compute Fiedler vector (second smallest eigenvector of L)
    # For n=200, full eigendecomposition is fast enough
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Sort by eigenvalue to be safe
        idx = np.argsort(eigenvalues)
        fiedler = eigenvectors[:, idx[1]]
    except np.linalg.LinAlgError:
        # Fallback to a simple heuristic if eigendecomposition fails
        fiedler = np.zeros(n)
    
    # Ensure deterministic sign
    if np.sum(fiedler) < 0:
        fiedler = -fiedler
    
    # Compute neighbor degree sum (second-order degree)
    neighbor_degree_sums = A @ degrees
    
    # Compute k-core numbers (approximate via iterative peeling)
    # This identifies the "core" structure of the graph
    core_numbers = np.zeros(n, dtype=int)
    remaining = np.ones(n, dtype=bool)
    current_degrees = degrees.copy().astype(int)
    
    while remaining.any():
        # Find minimum degree among remaining vertices
        min_deg = current_degrees[remaining].min()
        if min_deg <= 0:
            break
        
        # Assign core number to all vertices with this degree
        mask = remaining & (current_degrees == min_deg)
        core_numbers[mask] = min_deg
        
        # Remove these vertices and update degrees
        remaining[mask] = False
        # Update degrees of neighbors of removed vertices
        for v in np.where(mask)[0]:
            neighbors = np.where(A[v] > 0)[0]
            for nb in neighbors:
                if remaining[nb]:
                    current_degrees[nb] -= 1
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 120:
        # Return a simple fallback score
        return -degrees.astype(float)
    
    # Construct composite score:
    # 1. |Fiedler| values: high absolute values indicate peripheral/boundary vertices
    # 2. -degree: low degree is preferred (less blocking)
    # 3. neighbor_degree_sum: high neighbor degree sum means we remove valuable hubs
    # 4. -core_number: low core number means we're on the periphery, not in dense core
    
    # Normalize features to comparable scales
    fiedler_abs = np.abs(fiedler)
    fiedler_abs_norm = (fiedler_abs - fiedler_abs.min()) / (fiedler_abs.max() - fiedler_abs.min() + 1e-10)
    
    deg_norm = degrees / (degrees.max() + 1e-10)
    
    nds_norm = neighbor_degree_sums / (neighbor_degree_sums.max() + 1e-10)
    
    core_norm = core_numbers.astype(float) / (core_numbers.max() + 1e-10)
    
    # Weighted combination
    # Emphasize Fiedler (spectral separation) and low degree, with some neighbor degree sum
    weights = {
        'fiedler': 0.35,
        'neg_degree': 0.25,
        'neighbor_deg_sum': 0.25,
        'neg_core': 0.15
    }
    
    scores = (
        weights['fiedler'] * fiedler_abs_norm +
        weights['neg_degree'] * (1.0 - deg_norm) +
        weights['neighbor_deg_sum'] * nds_norm +
        weights['neg_core'] * (1.0 - core_norm)
    )
    
    # Add small random tiebreaker to ensure deterministic ordering for equal scores
    # Use vertex index as a deterministic tiebreaker
    scores += np.arange(n) * 1e-10
    
    return scores
# EVOLVE-BLOCK-END
