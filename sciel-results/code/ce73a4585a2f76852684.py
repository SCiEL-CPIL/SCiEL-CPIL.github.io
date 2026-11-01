def _als_decomposition(T, n, m, p, target_rank, time_budget=15):
    """Alternating Least Squares for tensor decomposition."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * p  # Wait, p*n = 20
    
    dim_u = n * m  # 4
    dim_v = m * p  # 20
    dim_w = p * n  # 20
    
    start_time = time.time()
    
    for trial in range(10):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(dim_u, target_rank)
        V = np.random.randn(dim_v, target_rank)
        W = np.random.randn(dim_w, target_rank)
        
        for step in range(100):
            if time.time() - start_time > time_budget:
                break
            
            # Update U: for each i, solve U[i,:] @ (V ⊙ W) = T[i,:,:].reshape(-1)
            # V ⊙ W is Khatri-Rao: column r is V[:,r] ⊗ W[:,r]
            VW = np.kron(V.T, W.T).T  # shape (dim_v * dim_w, target_rank)
            # Actually, np.kron(V.T, W.T) gives (target_rank * dim_v, target_rank * dim_w) which is wrong
            # Let me do it manually
            # Column r of VW should be V[:,r] ⊗ W[:,r]
            VW = np.zeros((dim_v * dim_w, target_rank))
            for r in range(target_rank):
                VW[:, r] = np.kron(V[:, r], W[:, r])
            
            for i in range(dim_u):
                rhs = T[i, :, :].reshape(-1)
                # Solve VW @ U[i,:] = rhs
                U[i, :], residuals, rank_vw, sv = np.linalg.lstsq(VW, rhs, rcond=None)
            
            # Update V
            UW = np.zeros((dim_u * dim_w, target_rank))
            for r in range(target_rank):
                UW[:, r] = np.kron(U[:, r], W[:, r])
            for j in range(dim_v):
                rhs = T[:, j, :].reshape(-1)
                V[j, :], residuals, rank_uw, sv = np.linalg.lstsq(UW, rhs, rcond=None)
            
            # Update W
            UV = np.zeros((dim_u * dim_v, target_rank))
            for r in range(target_rank):
                UV[:, r] = np.kron(U[:, r], V[:, r])
            for k in range(dim_w):
                rhs = T[:, :, k].reshape(-1)
                W[k, :], residuals, rank_uv, sv = np.linalg.lstsq(UV, rhs, rcond=None)
            
            # Check convergence
            T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
            loss = np.sum((T - T_est)**2)
            if loss < 1e-12:
                break
        
        T_final = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.max(np.abs(T - T_final)) < 1e-8:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None
