<<<<<<< SEARCH
    def try_rank(r, max_iters=500, seed=42):
        """Try to find a rank-r decomposition using alternating least squares."""
        if r < max(n * m, m * p, p * n):
            return None  # Impossible
        
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each column c, minimize ||T - sum_c' U_c'⊗V_c'⊗W_c'||
            # This is: for each c, U[:,c] should approximate the "residual" projected
            # Actually, use ALS: fix V,W, solve for U; fix U,W, solve for V; fix U,V, solve for W
            
            # Update U: T reshaped as (n*m, m*p*p*n) vs V⊗W
            # T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
            # For fixed V,W: T_reshaped[a, (b,c)] = sum_d U[a,d] * (V[b,d]*W[c,d])
            # So: T_2d = U @ (V⊗W).T where V⊗W is (r, m*p*p*n)
            VW = np.einsum('bd,cd->bcd', V, W)  # (m*p, p*n, r)
            VW_2d = VW.transpose(2, 0, 1).reshape(r, m * p * p * n)
            T_2d = T.reshape(n * m, m * p * p * n)
            # U @ VW_2d.T = T_2d => U = T_2d @ pinv(VW_2d.T)
            try:
                U_new = T_2d @ np.linalg.pinv(VW_2d.T)
            except:
                U_new = U
            U = U_new
            
            # Update V
            UW = np.einsum('ad,cd->acd', U, W)  # (n*m, p*n, r)
            UW_2d = UW.transpose(2, 0, 1).reshape(r, n * m * p * n)
            T_2d_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V_new = T_2d_v @ np.linalg.pinv(UW_2d.T)
            except:
                V_new = V
            V = V_new
            
            # Update W
            UV = np.einsum('ad,bd->abd', U, V)  # (n*m, m*p, r)
            UV_2d = UV.transpose(2, 0, 1).reshape(r, n * m * m * p)
            T_2d_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W_new = T_2d_w @ np.linalg.pinv(UV_2d.T)
            except:
                W_new = W
            W = W_new
            
            # Check error
            recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-8:
                break
        
        # Try rounding to half-integers
        U_h = to_half_int(best_U)
        V_h = to_half_int(best_V)
        W_h = to_half_int(best_W)
        
        if check_exact(U_h, V_h, W_h, T):
            return U_h, V_h, W_h
        
        # Try rounding the current (not just best)
        U_h2 = to_half_int(U)
        V_h2 = to_half_int(V)
        W_h2 = to_half_int(W)
        if check_exact(U_h2, V_h2, W_h2, T):
            return U_h2, V_h2, W_h2
        
        return None
=======
    def try_rank(r, max_iters=100, seed=42):
        """Try to find a rank-r decomposition using SVD-init ALS."""
        if r < max(n * m, m * p, p * n):
            return None  # Impossible
        
        np.random.seed(seed)
        
        # SVD Initialization
        T_1 = T.reshape(n * m, m * p * p * n)
        T_2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
        T_3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
        
        try:
            U_svd, _, _ = np.linalg.svd(T_1, full_matrices=False)
            V_svd, _, _ = np.linalg.svd(T_2, full_matrices=False)
            W_svd, _, _ = np.linalg.svd(T_3, full_matrices=False)
            
            U = U_svd[:, :r].copy()
            V = V_svd[:, :r].copy()
            W = W_svd[:, :r].copy()
            
            # Add small random perturbation
            U += np.random.randn(n * m, r) * 0.1
            V += np.random.randn(m * p, r) * 0.1
            W += np.random.randn(p * n, r) * 0.1
        except:
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)

        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            VW = np.einsum('bd,cd->bcd', V, W)
            VW_2d = VW.transpose(2, 0, 1).reshape(r, m * p * p * n)
            T_2d = T.reshape(n * m, m * p * p * n)
            try:
                U_new, _, _, _ = np.linalg.lstsq(VW_2d.T, T_2d.T, rcond=None)
                U = U_new.T
            except:
                pass
                
            # Update V
            UW = np.einsum('ad,cd->acd', U, W)
            UW_2d = UW.transpose(2, 0, 1).reshape(r, n * m * p * n)
            T_2d_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V_new, _, _, _ = np.linalg.lstsq(UW_2d.T, T_2d_v.T, rcond=None)
                V = V_new.T
            except:
                pass
                
            # Update W
            UV = np.einsum('ad,bd->abd', U, V)
            UV_2d = UV.transpose(2, 0, 1).reshape(r, n * m * m * p)
            T_2d_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W_new, _, _, _ = np.linalg.lstsq(UV_2d.T, T_2d_w.T, rcond=None)
                W = W_new.T
            except:
                pass
            
            recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if err < 1e-8:
                break
        
        # Try rounding to half-integers
        U_h = to_half_int(best_U)
        V_h = to_half_int(best_V)
        W_h = to_half_int(best_W)
        
        if check_exact(U_h, V_h, W_h, T):
            return U_h, V_h, W_h
        
        # Try rounding the current
        U_h2 = to_half_int(U)
        V_h2 = to_half_int(V)
        W_h2 = to_half_int(W)
        if check_exact(U_h2, V_h2, W_h2, T):
            return U_h2, V_h2, W_h2
        
        return None
>>>>>>> REPLACE

And update the main loop to use more seeds since `lstsq` is faster.

