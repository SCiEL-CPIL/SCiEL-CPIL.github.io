"""Priority function using Fiedler vector for Maximum Independent Set."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def priority(adjacency):
    """
    Score vertices using the Fiedler vector of the graph Laplacian.
    Vertices with large |Fiedler component| are at the periphery of the
    graph's main connected structure and are safer to pick for an
    independent set (less likely to fragment the remaining subgraph).
    """
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]

    # Ensure symmetry
    A = (A + A.T) / 2.0

    # Compute degree vector
    deg = A.sum(axis=1)

    # Build Laplacian L = D - A
    # For sparse efficiency, use scipy sparse
    L_sparse = csr_matrix(-A)
    D_sparse = csr_matrix(np.diag(deg))
    L = D_sparse + L_sparse

    # For small graphs or if sparse fails, fall back to dense
    try:
        if n <= 1:
            return np.zeros(n, dtype=np.float64)

        # We want the second smallest eigenvalue's eigenvector (Fiedler vector)
        # eigsh finds eigenvalues near 'sigma' or the smallest by default
        # k=2 to get the two smallest; index 1 is Fiedler
        # Use 'SM' for smallest magnitude, but for Laplacian smallest is 0,
        # so we shift to find the second smallest
        # Actually, for connected graphs, lambda_0 = 0, lambda_1 > 0
        # eigsh(L, k=2, which='SM') should give us lambda_0 and lambda_1

        # For disconnected graphs, there may be multiple zero eigenvalues.
        # The Fiedler vector is still the eigenvector corresponding to the
        # smallest positive eigenvalue.

        # Use shift-invert to find eigenvalues near 0, but we want the
        # second smallest. Let's just get k=2 smallest.
        if n < 3:
            # Trivial cases
            if n == 1:
                return np.array([0.0])
            # n == 2
            if A[0, 1] > 0:
                return np.array([1.0, -1.0])
            else:
                return np.array([1.0, 1.0])

        # For n >= 3, use eigsh
        # which='SM' finds smallest magnitude eigenvalues
        # For Laplacian, these are the smallest eigenvalues
        try:
            vals, vecs = eigsh(L, k=2, which='SM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            fiedler = vecs[:, 1]
        except Exception:
            # Fallback: use dense eigendecomposition
            L_dense = L.toarray()
            vals, vecs = np.linalg.eigh(L_dense)
            fiedler = vecs[:, 1]

        # The Fiedler vector can have arbitrary sign; use magnitude
        scores = np.abs(fiedler)

        # Add a small tie-breaking term based on degree (lower degree = higher priority)
        # This helps when Fiedler components are similar
        deg_norm = deg / (deg.max() + 1e-10)
        scores = scores - 0.1 * deg_norm

        return scores.astype(np.float64)

    except Exception:
        # Ultimate fallback: degree-based
        return -deg.astype(np.float64)
# EVOLVE-BLOCK-END
