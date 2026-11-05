"""Rewritten program: spectral + local-structure priority for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex. The greedy decoder picks vertices in descending
    score order, skipping neighbours of already-picked vertices.
    
    Strategy: Combine spectral (Fiedler vector magnitude) with local structural
    features (degree, mean neighbor degree) to identify 'safe' peripheral vertices
    that are cheap to include in the independent set.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]

    # Compute degree
    degree = A.sum(axis=1)

    # Compute mean neighbor degree for each vertex
    # For vertex i, mean neighbor degree = sum of degrees of neighbors / degree of i
    # Avoid division by zero for isolated vertices
    neighbor_degree_sum = A @ degree
    mean_neighbor_degree = np.where(degree > 0, neighbor_degree_sum / np.maximum(degree, 1), 0.0)

    # Spectral approach: Compute Fiedler vector (2nd smallest eigenvalue of Laplacian)
    # Laplacian L = D - A
    D = np.diag(degree)
    L = D - A

    try:
        # Use scipy if available for better eigenvalue computation
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        L_sparse = csr_matrix(L)
        # Find the two smallest eigenvalues/eigenvectors
        # k=2 because we want the 2nd smallest (Fiedler)
        # sigma=0 shifts to find smallest
        vals, vecs = eigsh(L_sparse, k=2, sigma=0, which='LM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Fiedler vector is the eigenvector for the 2nd smallest eigenvalue
        if vecs.shape[1] >= 2:
            fiedler = vecs[:, 1]
        else:
            fiedler = vecs[:, -1]
    except Exception:
        # Fallback to numpy if scipy fails
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            fiedler = vecs[:, 1] if vecs.shape[1] >= 2 else vecs[:, -1]
        except Exception:
            # Last resort: use zero vector
            fiedler = np.zeros(n)

    # Normalize fiedler vector
    fiedler_mag = np.abs(fiedler)
    fiedler_mag_norm = fiedler_mag / (np.max(fiedler_mag) + 1e-10)

    # Check time
    if time.time() > deadline:
        # Return a simple score if we're running out of time
        return -degree

    # Combine features:
    # 1. Low degree is good (fewer vertices blocked)
    # 2. High mean neighbor degree is good (blocked vertices were less valuable anyway)
    # 3. High Fiedler magnitude is good (peripheral, less central to connectivity)
    
    # Normalize each feature to [0, 1] range for combination
    deg_norm = degree / (np.max(degree) + 1e-10)
    mnd_norm = mean_neighbor_degree / (np.max(mean_neighbor_degree) + 1e-10)
    
    # We want: low degree, high mean neighbor degree, high fiedler magnitude
    # Score = -w1*deg_norm + w2*mnd_norm + w3*fiedler_mag_norm
    # Let's weight them: degree is most important, then fiedler, then mean neighbor degree
    w1 = 0.5
    w2 = 0.2
    w3 = 0.3
    
    score = -w1 * deg_norm + w2 * mnd_norm + w3 * fiedler_mag_norm

    # Check time again
    if time.time() > deadline:
        return score

    return score.astype(float)
# EVOLVE-BLOCK-END
