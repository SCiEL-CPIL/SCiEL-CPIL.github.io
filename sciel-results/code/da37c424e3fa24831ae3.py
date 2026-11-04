<<<<<<< SEARCH
def _gnn_embed(a, b, steps=3):
    """
    Compute node embeddings for facilities (from A) and locations (from B)
    using a simple message-passing mechanism (GNN-inspired).
    Returns embeddings of shape (n, d) where d = n for simplicity, or reduced.
    Here we use the matrix powers or repeated multiplication to capture higher-order structure.
    """
    n = len(a)
    
    # Normalize matrices to prevent explosion
    a_max = np.max(np.abs(a))
    b_max = np.max(np.abs(b))
    if a_max > 0:
        a_norm = a / a_max
    else:
        a_norm = a.copy()
    if b_max > 0:
        b_norm = b / b_max
    else:
        b_norm = b.copy()
    
    # Initialize embeddings as row sums (degree-like)
    e_f = a_norm.sum(axis=1)
    e_l = b_norm.sum(axis=1)
    
    # Message passing steps
    for _ in range(steps):
        # Update facility embeddings based on neighbor flows
        # e_f_new[i] = sum_j a_norm[i,j] * e_f[j]  (self-loop included)
        e_f = a_norm @ e_f
        # Update location embeddings based on neighbor distances
        e_l = b_norm @ e_l
        
    # Normalize embeddings
    e_f_norm = np.linalg.norm(e_f)
    e_l_norm = np.linalg.norm(e_l)
    if e_f_norm > 0:
        e_f = e_f / e_f_norm
    if e_l_norm > 0:
        e_l = e_l / e_l_norm
        
    return e_f, e_l

def _hungarian_initial(a, b):
    """
    Create an initial permutation using Hungarian algorithm on a linear surrogate.
    Combines row-sum centrality with GNN-inspired embeddings.
    Surrogate cost: C[i, j] = -w1 * a_row_sums[i] * b_row_sums[j] - w2 * dot(e_f[i], e_l[j])
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # GNN embeddings
    e_f, e_l = _gnn_embed(a, b, steps=2)
    
    # Construct surrogate cost matrix
    # We want to assign high a_row_sums to high b_row_sums
    # And high similarity between embeddings
    # Cost[i, j] = - (w1 * a_row_sums[i] * b_row_sums[j] + w2 * e_f[i] * e_l[j])
    
    # Normalize row sums for combination
    ars_max = np.max(np.abs(a_row_sums))
    brs_max = np.max(np.abs(b_row_sums))
    if ars_max > 0:
        ars_norm = a_row_sums / ars_max
    else:
        ars_norm = a_row_sums
    if brs_max > 0:
        brs_norm = b_row_sums / brs_max
    else:
        brs_norm = b_row_sums
        
    # Weighting
    w1 = 1.0
    w2 = 0.5
    
    # Outer product for row sums
    term1 = w1 * np.outer(ars_norm, brs_norm)
    
    # Dot product for embeddings (element-wise since embeddings are scalars here)
    term2 = w2 * np.outer(e_f, e_l)
    
    C = -(term1 + term2)
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    
    # row_ind is facilities, col_ind is locations
    # p[facility] = location
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p
=======
def _ql_initial(a, b, n_eigs=5):
    """
    Quadratic Linearization (QL) initialization:
    Decompose A into rank-one outer products of dominant eigenvectors.
    For each pair (u_k, v_k), solve a linear assignment minimizing sum a_ik * b_p[i],p[j].
    Combine multiple LAs to get a high-quality initial permutation.
    """
    n = len(a)
    
    # Eigendecomposition of A (symmetrize first for stability)
    A_sym = (a + a.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(A_sym)
    
    # Sort by absolute eigenvalue descending
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Use top n_eigs eigenvectors
    k = min(n_eigs, n)
    
    # Build surrogate cost matrix by combining rank-one decompositions
    # For eigenvector u_k with eigenvalue lambda_k:
    #   contribution to cost ~ lambda_k * sum_i,j u_k[i]*u_k[j]*b[p[i],p[j]]
    #   Linear surrogate: assign facility i to location j to minimize
    #   sum_k lambda_k * u_k[i] * (sum_j' u_k[j'] * b[j, p[j']])
    # Approximate: C[i,j] = sum_k lambda_k * u_k[i] * w_k[j]
    # where w_k[j] = sum_i u_k[i] * b[i,j] (projection of b onto u_k)
    
    C = np.zeros((n, n), dtype=np.float64)
    
    for k_idx in range(k):
        lam = eigenvalues[k_idx]
        u = eigenvectors[:, k_idx]
        
        # Weight vector for locations: w[j] = sum_i u[i] * b[i, j]
        w = u @ b  # shape (n,)
        
        # Surrogate cost contribution: -lam * u[i] * w[j]
        # We want to minimize total cost, so negative for high alignment
        C += -lam * np.outer(u, w)
    
    # Also add row-sum based term for stability
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    ars_max = np.max(np.abs(a_row_sums))
    brs_max = np.max(np.abs(b_row_sums))
    if ars_max > 0 and brs_max > 0:
        C += -0.1 * np.outer(a_row_sums / ars_max, b_row_sums / brs_max)
    
    # Solve the linear assignment
    row_ind, col_ind = linear_sum_assignment(C)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _hungarian_initial(a, b):
    """
    Create an initial permutation using QL-based surrogate + Hungarian.
    """
    return _ql_initial(a, b, n_eigs=5)
>>>>>>> REPLACE

Now let me add a 3-opt polish function and update the solve function:

