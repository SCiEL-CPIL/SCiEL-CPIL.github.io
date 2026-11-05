"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Compute vertex scores for greedy Maximum Independent Set selection.
    
    Strategy: Use a combination of the Fiedler vector (second-smallest eigenvector
    of the Laplacian) and degree-based features. The Fiedler vector identifies
    vertices that lie on natural graph cuts / are minimally entangled in the
    connectivity structure. Vertices with larger |fiedler| are at boundaries
    between components and are good candidates for early selection.
    
    We combine:
    - Absolute Fiedler vector value (prefer nodes at natural cuts / boundaries)
    - Negative degree (prefer low-degree nodes)
    - A bonus for nodes whose neighbors have high degree (prefer leaves/pendants)
    - Negative mean neighbor degree (prefer nodes not deeply embedded in dense core)
    
    The greedy decoder picks vertices in descending score order, skipping neighbors
    of already-picked vertices.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    # Handle trivial cases
    if n == 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([0.0], dtype=np.float64)
    
    # Ensure symmetric
    A = 0.5 * (A + A.T)
    
    # Degree vector
    deg = A.sum(axis=1)
    
    # Compute Fiedler vector: second-smallest eigenvector of the graph Laplacian
    # L = D - A
    # We use power iteration on a shifted/scaled Laplacian to find the second eigenvector,
    # or more practically, we compute a few eigenvectors via the Lanczos/Arnoldi method
    # using scipy if available, otherwise fall back to power iteration with deflation.
    
    # Use scipy.sparse.linalg.eigsh for the second-smallest eigenvalue
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        import scipy.sparse as sp
        
        A_sp = csr_matrix(A)
        D_sp = sp.diags(deg)
        L_sp = D_sp - A_sp
        
        # eigsh finds k eigenvalues; we want the second-smallest (index 1, since 0 is the smallest)
        # The smallest eigenvalue of L is 0 (for connected graph)
        # We want eigenvalue index 1 (the Fiedler value)
        if n > 2:
            k = min(3, n - 1)
            vals, vecs = eigsh(L_sp, k=k, which='SM')  # smallest magnitude
            # Sort by eigenvalue
            idx = np.argsort(vals)
            # The Fiedler vector is the one corresponding to the second-smallest eigenvalue
            # vals[0] should be ~0, vals[1] is the Fiedler value
            fiedler_idx = 1
            if fiedler_idx < len(idx):
                fiedler = vecs[:, idx[fiedler_idx]]
            else:
                fiedler = np.zeros(n)
        else:
            fiedler = np.zeros(n)
        
        # Ensure consistent sign (the Fiedler vector can be flipped)
        if np.sum(fiedler) < 0:
            fiedler = -fiedler
        
    except Exception:
        # Fallback: use power iteration to approximate the Fiedler vector
        # L = D - A; we want the second eigenvector
        # Use inverse iteration with shift, or compute via Rayleigh quotient iteration
        # Simpler fallback: use the principal eigenvector of (L - sigma*I) for some sigma
        # Or just use the absolute value of a random projection after deflating the first mode
        
        D = np.diag(deg)
        L = D - A
        
        # Deflate: find first eigenvector (all-ones normalized), then find second
        v1 = np.ones(n, dtype=np.float64)
        v1 = v1 / np.linalg.norm(v1)
        
        # Power iteration on L for the smallest non-zero eigenvalue is tricky
        # Instead, use the fact that for the Fiedler vector, we can use
        # the eigenvector of L orthogonal to 1 with smallest Rayleigh quotient
        # Approximate via a few steps of inverse iteration with random start orthogonal to 1
        
        # Random start orthogonal to 1
        v = np.random.randn(n)
        v = v - np.mean(v) * np.ones(n)
        v = v / np.linalg.norm(v)
        
        for it in range(50):
            if time.time() > deadline:
                break
            # Solve L @ w = v (use conjugate gradient if possible, else approximate)
            try:
                # Use np.linalg.lstsq for small n, or iterative
                # For n=200, direct solve is fine
                w, residuals, rank, sv = np.linalg.lstsq(L, v, rcond=None)
            except Exception:
                w = v  # fallback
            w = w - np.mean(w) * np.ones(n)  # re-orthogonalize
            nw = np.linalg.norm(w)
            if nw < 1e-15:
                break
            w = w / nw
            if np.linalg.norm(w - v) < 1e-10:
                v = w
                break
            v = w
        
        fiedler = v
    
    # Absolute value of Fiedler vector: larger |fiedler| = more "boundary" / cut node
    abs_fiedler = np.abs(fiedler)
    
    # Feature: min neighbor degree
    min_nbr_deg = np.full(n, 0.0)
    for i in range(n):
        if time.time() > deadline:
            break
        nbrs = np.where(A[i] > 0.5)[0]
        if len(nbrs) > 0:
            min_nbr_deg[i] = np.min(deg[nbrs])
    
    # Feature: mean neighbor degree
    mean_nbr_deg = np.zeros(n)
    for i in range(n):
        if time.time() > deadline:
            break
        nbrs = np.where(A[i] > 0.5)[0]
        if len(nbrs) > 0:
            mean_nbr_deg[i] = np.mean(deg[nbrs])
    
    # Normalize features
    def normalize(x):
        x = np.asarray(x, dtype=np.float64)
        mn = np.min(x)
        mx = np.max(x)
        if mx - mn < 1e-15:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)
    
    deg_norm = normalize(deg)
    fiedler_norm = normalize(abs_fiedler)
    min_nbr_norm = normalize(min_nbr_deg)
    mean_nbr_norm = normalize(mean_nbr_deg)
    
    # Combine:
    # +fiedler: prefer nodes at natural cuts (large |Fiedler value|)
    # -degree: prefer low degree
    # +min_nbr_deg: prefer pendant nodes (neighbors in dense core)
    # -mean_nbr_deg: prefer nodes not deeply in dense neighborhood
    
    w1 = 1.8   # weight for +|Fiedler| (global cut structure)
    w2 = 1.0   # weight for -degree
    w3 = 0.7   # weight for +min neighbor degree
    w4 = 0.3   # weight for -mean neighbor degree
    
    scores = (w1 * fiedler_norm) + (-w2 * deg_norm) + (w3 * min_nbr_norm) + (-w4 * mean_nbr_norm)
    
    return scores.astype(np.float64)
# EVOLVE-BLOCK-END
