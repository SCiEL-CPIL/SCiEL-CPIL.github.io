    def als_decompose(rank, seed=0):
        """Run ALS with SVD initialization."""
        rng = np.random.RandomState(seed)
        
        # SVD Initialization
        U = None
        V = None
        W = None
        
        try:
            U_svd, _, _ = np.linalg.svd(T_flat, full_matrices=False)
            if U_svd.shape[1] >= rank:
                U = U_svd[:, :rank].copy()
            else:
                # Pad with random vectors
                U = np.zeros((U_size, rank))
                U[:, :U_svd.shape[1]] = U_svd
                U[:, U_svd.shape[1]:] = rng.randn(U_size, rank - U_svd.shape[1])
        except:
            U = rng.randn(U_size, rank)
            
        try:
            V_svd, _, _ = np.linalg.svd(T_flat_v, full_matrices=False)
            if V_svd.shape[1] >= rank:
                V = V_svd[:, :rank].copy()
            else:
                V = np.zeros((V_size, rank))
                V[:, :V_svd.shape[1]] = V_svd
                V[:, V_svd.shape[1]:] = rng.randn(V_size, rank - V_svd.shape[1])
        except:
            V = rng.randn(V_size, rank)
            
        try:
            W_svd, _, _ = np.linalg.svd(T_flat_w, full_matrices=False)
            if W_svd.shape[1] >= rank:
                W = W_svd[:, :rank].copy()
            else:
                W = np.zeros((W_size, rank))
                W[:, :W_svd.shape[1]] = W_svd
                W[:, W_svd.shape[1]:] = rng.randn(W_size, rank - W_svd.shape[1])
        except:
            W = rng.randn(W_size, rank)
        
        # Scale initialization to avoid zero gradients
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_cost = np.inf
        for it in range(200):
            if time.time() > deadline:
                return None, np.inf
            
            # Update U
            M_vw = np.einsum('vc,wc->vwc', V, W).reshape(V_size * W_size, rank)
            U_new, _, _, _ = np.linalg.lstsq(M_vw.T, T_flat.T, rcond=None)
            U = U_new.T
            
            # Update V
            M_uw = np.einsum('uc,wc->uwc', U, W).reshape(U_size * W_size, rank)
            V_new, _, _, _ = np.linalg.lstsq(M_uw.T, T_flat_v.T, rcond=None)
            V = V_new.T
            
            # Update W
            M_uv = np.einsum('uc,vc->uvc', U, V).reshape(U_size * V_size, rank)
            W_new, _, _, _ = np.linalg.lstsq(M_uv.T, T_flat_w.T, rcond=None)
            W = W_new.T
            
            # Cost calculation
            T_calc = np.einsum('uc,vc,wc->uvw', U, V, W)
            cost = np.sum((T - T_calc)**2)
            
            if abs(prev_cost - cost) < 1e-12:
                break
            prev_cost = cost
            
        T_calc = np.einsum('uc,vc,wc->uvw', U, V, W)
        resid = np.max(np.abs(T_calc - T))
        return (U, V, W), resid
