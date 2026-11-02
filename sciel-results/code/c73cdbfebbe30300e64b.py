# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    # Build the multiplication tensor T of shape (p*n, n*m, m*p)
    # T[(k*n+i), (i*m+j), (j*p+k)] = 1
    T = np.zeros((p * n, n * m, m * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[k * n + i, i * m + j, j * p + k] = 1.0
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook()
    best_rank = n * m * p
    
    def verify(U_mat, V_mat, W_mat, atol=1e-10):
        """Verify that sum_r W[:,r] \otimes U[:,r] \otimes V[:,r] == T"""
        T_recon = np.einsum('ar,br,cr->abc', W_mat, U_mat, V_mat)
        return np.allclose(T_recon, T, atol=atol)

    def greedy_init(R, T_tensor):
        """
        Initialize U, V, W using greedy rank-1 extraction.
        """
        U_init = np.zeros((n * m, R))
        V_init = np.zeros((m * p, R))
        W_init = np.zeros((p * n, R))
        T_res = T_tensor.copy()
        
        for r in range(R):
            if time.time() > deadline - 10:
                break
                
            # Power method for best rank-1 approximation
            u = np.random.randn(n * m)
            v = np.random.randn(m * p)
            w = np.random.randn(p * n)
            
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
            w /= np.linalg.norm(w)
            
            for _ in range(100): # Max power iterations
                w_new = np.einsum('abc,b,c->a', T_res, u, v)
                v_new = np.einsum('abc,b,a->c', T_res, u, w_new)
                u_new = np.einsum('abc,a,c->b', T_res, w, v_new)
                
                nu = np.linalg.norm(u_new)
                nv = np.linalg.norm(v_new)
                nw = np.linalg.norm(w_new)
                
                if nu > 1e-15: u_new /= nu
                if nv > 1e-15: v_new /= nv
                if nw > 1e-15: w_new /= nw
                
                if np.linalg.norm(w_new - w) < 1e-10:
                    break
                    
                u, v, w = u_new, v_new, w_new
                
            U_init[:, r] = u
            V_init[:, r] = v
            W_init[:, r] = w
            
            T_res -= np.einsum('b,c,a->abc', u, v, w)
            
        return U_init, V_init, W_init

    def try_rank(R, max_restarts=50, max_iters_per_restart=200):
        """Try to find exact rank-R decomposition using greedy init + ALS refinement."""
        if R >= best_rank:
            return None
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            U_r, V_r, W_r = greedy_init(R, T)
            
            # Check if greedy alone is sufficient
            T_recon = np.einsum('ar,br,cr->abc', W_r, U_r, V_r)
            loss = np.linalg.norm(T_recon - T)
            if loss < 1e-10:
                return U_r, V_r, W_r

            prev_loss = loss
            converged = False
            
            for it in range(max_iters_per_restart):
                if time.time() > deadline - 10:
                    break
                
                # Optimize W
                VO = np.einsum('cr,br->bcr', V_r, U_r)
                VO_flat = VO.reshape(-1, R)
                T_flat = T.reshape(p * n, -1)
                
                W_new = np.zeros((p * n, R))
                for a in range(p * n):
                    try:
                        x, _, _, _ = np.linalg.lstsq(VO_flat, T_flat[a, :], rcond=None)
                        W_new[a, :] = x
                    except np.linalg.LinAlgError:
                        pass
                
                # Optimize U
                WV = np.einsum('ar,cr->acr', W_new, V_r)
                WV_flat = WV.reshape(-1, R)
                
                U_new = np.zeros((n * m, R))
                for b in range(n * m):
                    T_b = T[:, b, :].reshape(-1)
                    try:
                        x, _, _, _ = np.linalg.lstsq(WV_flat, T_b, rcond=None)
                        U_new[b, :] = x
                    except np.linalg.LinAlgError:
                        pass
                
                # Optimize V
                UW = np.einsum('br,ar->abr', U_new, W_new)
                M_V = UW.swapaxes(0, 1).reshape(-1, R)
                
                V_new = np.zeros((m * p, R))
                for c in range(m * p):
                    T_c = T[:, :, c].reshape(-1)
                    try:
                        x, _, _, _ = np.linalg.lstsq(M_V, T_c, rcond=None)
                        V_new[c, :] = x
                    except np.linalg.LinAlgError:
                        pass
                
                U_r, V_r, W_r = U_new, V_new, W_new
                
                T_recon = np.einsum('ar,br,cr->abc', W_r, U_r, V_r)
                loss = np.linalg.norm(T_recon - T)
                
                if loss < 1e-10:
                    converged = True
                    break
                
                if abs(prev_loss - loss) < 1e-15:
                    break
                prev_loss = loss
            
            if converged:
                return U_r, V_r, W_r
        
        return None
    
    for R in range(41, 20, -1):
        if time.time() > deadline - 20:
            break
        result = try_rank(R, max_restarts=20, max_iters_per_restart=100)
        if result is not None:
            U, V, W = result
            best_rank = R
            break
    
    if not verify(U, V, W):
        U, V, W = schoolbook()
        best_rank = n * m * p
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
