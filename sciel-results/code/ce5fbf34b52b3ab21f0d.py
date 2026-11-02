def try_rank_vectorized(r, T, max_iters=500, timeout=140.0, seed=0):
    """
    Try to find a rank-r decomposition using vectorized ALS.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    U = rng.randn(N, r)
    V = rng.randn(M, r)
    W = rng.randn(P, r)
    
    # Precompute T reshapes
    # T is (N, M, P)
    T_U = T.reshape(N, M * P)
    T_V = T.transpose(1, 0, 2).reshape(M, N * P)
    T_W = T.transpose(2, 0, 1).reshape(P, N * M)
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() - start_time > timeout - 5:
            break
        
        # Update U
        # T_U = U @ (V * W).T ? 
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # T_U[a, b*M+c] = sum_r U[a, r] V[b, r] W[c, r]
        # Let K_WV = W ⊗ V. Shape (M*P, r).
        # (K_WV)[b*M+c, r] = W[c, r] * V[b, r]
        # Then (U @ K_WV.T)[a, b*M+c] = sum_r U[a, r] * K_WV.T[r, b*M+c] = sum_r U[a, r] V[b, r] W[c, r]
        # So T_U ≈ U @ K_WV.T
        # We want U = argmin || T_U - U K_WV.T ||
        # This is U = T_U @ (K_WV.T @ K_WV)^{-1} @ K_WV.T
        K_WV = np.kron(W, V) # (M*P, r)
        A_U = K_WV.T # (r, M*P)
        try:
            # Solve U A_U.T = T_U
            # U = T_U @ inv(A_U.T @ A_U) @ A_U.T
            # A_U.T @ A_U is (M*P, M*P) ?? No.
            # A_U is (r, M*P). A_U.T is (M*P, r).
            # We want U (N, r) such that U @ A_U.T (N, M*P) = T_U (N, M*P).
            # This is underdetermined if r < M*P.
            # Least squares solution: U = T_U @ (A_U.T @ A_U)^+ ?
            # Let X = A_U.T. We want U X = T_U.
            # X is (M*P, r).
            # U = T_U @ X.T @ (X @ X.T)^{-1}.
            # X @ X.T is (M*P, M*P). This is large (300x300).
            # Alternatively, use lstsq on the transposed system?
            # T_U.T (M*P, N) = X (M*P, r) @ U.T (r, N).
            # So we solve for U.T: U.T = lstsq(X, T_U.T).
            U_T = np.linalg.lstsq(K_WV, T_U.T, rcond=None)[0]
            U = U_T.T
        except:
            pass
        
        # Update V
        # T_V[b, a*N+c] = sum_r V[b, r] U[a, r] W[c, r]
        # Let K_UW = U ⊗ W. Shape (N*P, r).
        # (K_UW)[a*N+c, r] = U[a, r] * W[c, r]
        # (V @ K_UW.T)[b, a*N+c] = sum_r V[b, r] U[a, r] W[c, r]
        # T_V ≈ V @ K_UW.T
        # V.T = lstsq(K_UW, T_V.T)
        K_UW = np.kron(U, W) # (N*P, r)
        try:
            V_T = np.linalg.lstsq(K_UW, T_V.T, rcond=None)[0]
            V = V_T.T
        except:
            pass
        
        # Update W
        # T_W[c, a*N+b] = sum_r W[c, r] U[a, r] V[b, r]
        # Let K_UV = U ⊗ V. Shape (N*M, r).
        # (K_UV)[a*N+b, r] = U[a, r] * V[b, r]
        # (W @ K_UV.T)[c, a*N+b] = sum_r W[c, r] U[a, r] V[b, r]
        # T_W ≈ W @ K_UV.T
        # W.T = lstsq(K_UV, T_W.T)
        K_UV = np.kron(U, V) # (N*M, r)
        try:
            W_T = np.linalg.lstsq(K_UV, T_W.T, rcond=None)[0]
            W = W_T.T
        except:
            pass
        
        # Compute error
        result = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(result - T)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if error < 1e-8:
                # Snap to nearest half-integers
                best_U = np.round(best_U * 2) / 2
                best_V = np.round(best_V * 2) / 2
                best_W = np.round(best_W * 2) / 2
                if verify_decomposition(best_U, best_V, best_W, T):
                    return best_U, best_V, best_W
    
    if best_error < 1e-8:
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        if verify_decomposition(best_U, best_V, best_W, T):
            return best_U, best_V, best_W
    
    return None
