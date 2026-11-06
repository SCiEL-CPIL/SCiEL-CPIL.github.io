import numpy as np
import time
from scipy.linalg import eigh

def priority(adjacency):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s / 150s kill

    n = adjacency.shape[0]
    A = np.asarray(adjacency, dtype=bool)
    A_float = A.astype(float)

    deg = A.sum(axis=1).astype(float)
    has_edge = deg > 0

    # Isolated vertices: always safe to pick -> top priority
    BIG = 1e9

    # Fixed-point iteration for "safe-to-pick" score
    # s[i] = 1 / (1 + sum_{j in nbr(i)} s[j])
    s = 1.0 / (deg + 1.0).copy()
    s[~has_edge] = BIG

    for it in range(200):
        if time.time() > deadline:
            break
        nbr_sum = A_float @ s
        new_s = 1.0 / (1.0 + nbr_sum)
        new_s[~has_edge] = BIG
        if np.allclose(new_s, s, rtol=1e-6, atol=1e-8):
            s = new_s
            break
        s = new_s

    # Spectral refinement: compute Fiedler vector (2nd smallest eigenvector of Laplacian)
    # Vertices with large |Fiedler component| are at the periphery and safer to include
    if n > 1 and time.time() < deadline - 5.0:
        # Build Laplacian L = D - A
        D = np.diag(deg)
        L = D - A_float
        
        # For efficiency, use scipy's eigh with subset selection
        # We want the eigenvector corresponding to the 2nd smallest eigenvalue
        try:
            # Use subset_by_index to only compute the first 2 eigenpairs
            # This is much faster than computing all eigenvalues
            evals, evecs = eigh(L, subset_by_index=[1, 1], check_finite=False)
            fiedler = evecs[:, 0]
            
            # Use magnitude of Fiedler vector as a boost for peripheral vertices
            # Normalize fiedler to [0, 1] range
            fiedler_abs = np.abs(fiedler)
            if fiedler_abs.max() > 0:
                fiedler_norm = fiedler_abs / fiedler_abs.max()
                # Add spectral component: peripheral vertices (high fiedler value) get a boost
                # This helps identify vertices that are less central to connectivity
                spectral_weight = 0.1
                s[has_edge] += spectral_weight * fiedler_norm[has_edge]
        except Exception:
            pass  # Fall back to fixed-point scores only if spectral computation fails

    # Final score: higher = pick earlier
    scores = s.copy()

    # Add a small tie-breaker based on degree (lower degree slightly preferred)
    deg_rank = -deg / (deg.max() + 1) if deg.max() > 0 else 0.0
    scores = scores + 1e-6 * deg_rank

    if time.time() > deadline:
        return scores

    return scores
