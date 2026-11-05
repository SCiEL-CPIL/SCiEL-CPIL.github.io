def _verify_decomposition(U, V, W, n=6, m=6, p=6, deadline=140.0):
    """
    Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    t0 = time.time()
    rank = U.shape[1]
    max_err = 0.0
    
    # Iterate over k and j to keep memory low, vectorize over i
    for k in range(p):
        # W[k*n + i, :] for i in 0..n-1
        # Rows are k*n, k*n+1, ..., k*n+n-1
        W_block = W[k*n : (k+1)*n, :]  # Shape (n, rank)
        
        for j in range(m):
            # V[j*p + k, :]
            V_vec = V[j*p + k, :]  # Shape (rank,)
            
            # U[i*m + j, :] for i in 0..n-1
            # Rows are j, j+m, j+2m, ...
            U_block = U[j::m, :]  # Shape (n, rank)
            
            # Compute element-wise product and sum over rank
            # (n, rank) * (n, rank) * (1, rank) -> (n, rank)
            # Sum over axis 1 -> (n,)
            sums = (U_block * W_block * V_vec).sum(axis=1)
            
            err = np.max(np.abs(sums - 1.0))
            if err > max_err:
                max_err = err
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return max_err
