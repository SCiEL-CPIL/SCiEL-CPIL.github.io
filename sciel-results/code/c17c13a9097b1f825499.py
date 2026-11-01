def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors."""
    R = U.shape[1]
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Reshape for vectorization
    # U reshaped to (n, m, R)
    U_r = U.reshape(n, m, R)
    V_r = V.reshape(m, p, R)
    W_r = W.reshape(p, n, R)
    
    # For each (i,j,k), compute sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # This is a batched operation
    
    # Expand dimensions for broadcasting
    # U_r: (n, m, 1, R)
    # V_r: (1, m, p, R)
    # W_r: (p, n, 1, R)
    
    # We want: for each (i,j,k), sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    
    # Let's do it step by step
    # First compute U_r[i,j,r] * V_r[j,k,r] for all (i,j,k,r)
    # U_r[:, :, None, :] * V_r[None, :, :, :]  # (n, m, p, R)
    
    prod_UV = U_r[:, :, None, :] * V_r[None, :, :, :]  # (n, m, p, R)
    
    # Now multiply by W_r[k,i,r]
    # W_r has shape (p, n, R), we need W_r[k,i,r] for each (i,j,k,r)
    # So we need to permute W_r to (n, p, R) and then expand
    W_perm = W_r.T  # (n, p, R)
    # W_perm[i,k,r] = W_r[k,i,r]
    
    prod_UVW = prod_UV * W_perm[:, None, :, :]  # (n, m, p, R)
    
    # Sum over R
    result = prod_UVW.sum(axis=3)  # (n, m, p)
    
    # Compute error
    error = np.sum((result - 1.0) ** 2)
    
    return error, result
