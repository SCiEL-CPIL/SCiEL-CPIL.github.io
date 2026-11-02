def _find_linear_dependence(U_arr, V_arr, W_arr, n, m, p, max_time=30):
    """
    Find a non-trivial linear dependence among the rank-1 terms.
    Returns (coefficients, indices_to_remove) or None.
    """
    start_time = time.time()
    r = U_arr.shape[1]
    
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    
    # Build the matrix of rank-1 terms
    # Each term r contributes to entry (i,j,k) the value U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
    # We want to find c such that sum_r c_r * term_r = 0
    # This is a system of n*m*p equations in r unknowns
    
    # For efficiency, build the matrix incrementally
    # M[i,j,k, r] = U3[i,j,r]*V3[j,k,r]*W3[k,i,r]
    
    # But this is n*m*p x r = 36 x r matrix
    # For r=32, this is 36x32, which is small
    
    M = np.zeros((n*m*p, r))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r_idx in range(r):
                    M[i*m*p + j*p + k, r_idx] = U3[i,j,r_idx] * V3[j,k,r_idx] * W3[k,i,r_idx]
    
    # Find null space of M
    # If null space is non-trivial, we have a linear dependence
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Find the dimension of the null space
    tol = 1e-10
    null_dim = sum(1 for sv in s if sv < tol)
    
    if null_dim == 0:
        return None
    
    # The null space vectors are the last null_dim columns of Vt
    null_vectors = Vt[-null_dim:]
    
    # For each null vector, find the term with the largest coefficient to eliminate
    best_elimination = None
    best_coeff = None
    
    for nv in null_vectors:
        # nv[r] is the coefficient for term r
        # We want to find a term with |nv[r]| > 0 to eliminate
        # After elimination, the new coefficients are:
        # new_coeff[r] = 1 + nv[r]/nv[elim_idx] for r != elim_idx
        # new_coeff[elim_idx] = 0
        
        # Choose the term with the largest |nv[r]| to minimize numerical issues
        max_idx = np.argmax(np.abs(nv))
        if abs(nv[max_idx]) < 1e-10:
            continue
        
        # Check if the elimination is valid
        # After elimination, term max_idx is removed
        # The remaining terms get scaled by (1 + nv[r]/nv[max_idx])
        
        # Verify that the new decomposition is valid
        new_U = U_arr.copy()
        new_V = V_arr.copy()
        new_W = W_arr.copy()
        
        for r_idx in range(r):
            if r_idx == max_idx:
                continue
            scale = 1.0 + nv[r_idx] / nv[max_idx]
            new_U[:, r_idx] *= scale
            new_V[:, r_idx] *= scale
            new_W[:, r_idx] *= scale
        
        # Remove the eliminated term
        keep = [i for i in range(r) if i != max_idx]
        new_U = new_U[:, keep]
        new_V = new_V[:, keep]
        new_W = new_W[:, keep]
        
        # Verify
        U3_new = new_U.reshape(n, m, -1)
        V3_new = new_V.reshape(m, p, -1)
        W3_new = new_W.reshape(p, n, -1)
        
        T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U3_new, V3_new, W3_new)
        T_true = np.ones((n, m, p))
        
        if np.allclose(T_reconstructed, T_true, atol=1e-10):
            if best_elimination is None or len(keep) < len(best_elimination[0]):
                best_elimination = (new_U, new_V, new_W)
                best_coeff = nv
    
    if best_elimination is not None:
        return best_elimination
    
    return None
