def _als_4x4x4(target_rank=49, max_iters=200, seed=42):
    """Use ALS to find a rank-r decomposition of the 4x4x4 multiplication tensor."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    n = 4
    T = np.ones((n, n, n))
    
    for r in range(target_rank, 65):
        U = np.random.randn(n*n, r) * 0.1
        V = np.random.randn(n*n, r) * 0.1
        W = np.random.randn(n*n, r) * 0.1
        
        for it in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(n):
                    M = (V[j*n:(j+1)*n, :] * W[:, i*n:(i+1)*n].T).T  # This is wrong
                    # Let me re-derive
                    # T[i,j,k] = sum_t U[i*n+j, t] * V[j*n+k, t] * W[k*n+i, t]
                    # For fixed (i,j): sum_t U[i*n+j, t] * (V[j*n+k, t] * W[k*n+i, t]) = T[i,j,k] for all k
                    # Let M[k, t] = V[j*n+k, t] * W[k*n+i, t]
                    # Then sum_t U[i*n+j, t] * M[k, t] = T[i,j,k] for all k
                    # So U[i*n+j, :] @ M.T = T[i,j, :]
                    M = V[j*n:(j+1)*n, :] * W[:, i*n:(i+1)*n].T  # (n, r) * (r, n) -> wrong dimensions
                    # V[j*n:(j+1)*n, :] is (n, r)
                    # W[:, i*n:(i+1)*n] is (r, n)
                    # We need M[k, t] = V[j*n+k, t] * W[k*n+i, t]
                    # So M = V[j*n:(j+1)*n, :] * W[:, i*n:(i+1)*n].T ... no
                    # W[:, i*n:(i+1)*n] is (r, n) with W[t, k] = W[k*n+i, t]
                    # So W[:, i*n:(i+1)*n].T is (n, r) with entry [k, t] = W[k*n+i, t]
                    # And V[j*n:(j+1)*n, :] is (n, r) with entry [k, t] = V[j*n+k, t]
                    # So M = V[j*n:(j+1)*n, :] * W[:, i*n:(i+1)*n].T is (n, r) with entry [k, t] = V[j*n+k, t] * W[k*n+i, t]
                    M = V[j*n:(j+1)*n, :] * W[:, i*n:(i+1)*n].T
                    # Now solve U[i*n+j, :] @ M.T = T[i,j, :]
                    # U[i*n+j, :] = T[i,j, :] @ M @ (M.T @ M)^{-1}
                    try:
                        U[i*n+j, :] = np.linalg.lstsq(M.T, T[i,j,:], rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(n):
                for k in range(n):
                    # T[i,j,k] = sum_t U[i*n+j, t] * V[j*n+k, t] * W[k*n+i, t]
                    # For fixed (j,k): sum_t (U[i*n+j, t] * W[k*n+i, t]) * V[j*n+k, t] = T[i,j,k] for all i
                    # Let M[i, t] = U[i*n+j, t] * W[k*n+i, t]
                    # M = U[:, j*n:(j+1)*n] * W[k*n:(k+1)*n, :].T ... 
                    # U[:, j*n:(j+1)*n] is (n, r) with entry [i, t] = U[i*n+j, t]
                    # W[k*n:(k+1)*n, :] is (n, r) with entry [i, t] = W[k*n+i, t]
                    # So M = U[:, j*n:(j+1)*n] * W[k*n:(k+1)*n, :] is (n, r) with entry [i, t] = U[i*n+j, t] * W[k*n+i, t]
                    M = U[:, j*n:(j+1)*n] * W[k*n:(k+1)*n, :]
                    # Solve M.T @ V[j*n+k, :] = T[:,j,k]
                    try:
                        V[j*n+k, :] = np.linalg.lstsq(M.T, T[:,j,k], rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(n):
                for i in range(n):
                    # T[i,j,k] = sum_t U[i*n+j, t] * V[j*n+k, t] * W[k*n+i, t]
                    # For fixed (k,i): sum_t (U[i*n+j, t] * V[j*n+k, t]) * W[k*n+i, t] = T[i,j,k] for all j
                    # Let M[j, t] = U[i*n+j, t] * V[j*n+k, t]
                    # M = U[i*n:(i+1)*n, :] * V[:, k*n:(k+1)*n].T ...
                    # U[i*n:(i+1)*n, :] is (n, r) with entry [j, t] = U[i*n+j, t]
                    # V[:, k*n:(k+1)*n] is (n, r) with entry [j, t] = V[j*n+k, t]
                    # So M = U[i*n:(i+1)*n, :] * V[:, k*n:(k+1)*n] is (n, r) with entry [j, t] = U[i*n+j, t] * V[j*n+k, t]
                    M = U[i*n:(i+1)*n, :] * V[:, k*n:(k+1)*n]
                    # Solve M.T @ W[k*n+i, :] = T[i,:,k]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M.T, T[i,:,k], rcond=None)[0]
                    except:
                        pass
        
        # Round to integers
        U_int = np.round(U).astype(int)
        V_int = np.round(V).astype(int)
        W_int = np.round(W).astype(int)
        
        # Verify
        max_err = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    val = np.dot(U_int[i*n+j], V_int[j*n+k] * W_int[k*n+i])
                    max_err = max(max_err, abs(val - 1))
        
        if max_err < 1e-6:
            return U_int.tolist(), V_int.tolist(), W_int.tolist(), r
    
    return None
