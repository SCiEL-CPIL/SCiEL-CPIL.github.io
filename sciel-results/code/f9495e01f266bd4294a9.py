<<<<<<< SEARCH
def _try_rank17_als(T, deadline):
    """
    Try to find a rank-17 decomposition using ALS with multiple initializations.
    Start from the rank-18 solution and try to merge/replace one term.
    """
    n, m, p = DIMENSIONS
    target_rank = 17
    
    U_struct, V_struct, W_struct = _build_structured_decomposition()
    
    # Try different strategies for initializing the rank-17 search
    strategies = []
    
    # Strategy 1: Remove the last schoolbook term and re-solve
    for seed in range(50):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize with rank-18 solution, zero out one column, add noise
        U = U_struct.copy()
        V = V_struct.copy()
        W = W_struct.copy()
        
        # Remove one column (try different ones)
        remove_col = seed % 18
        
        U_new = np.zeros((n * m, target_rank), dtype=np.float64)
        V_new = np.zeros((m * p, target_rank), dtype=np.float64)
        W_new = np.zeros((p * n, target_rank), dtype=np.float64)
        
        # Copy all columns except the removed one
        idx = 0
        for i in range(18):
            if i != remove_col:
                U_new[:, idx] = U[:, i]
                V_new[:, idx] = V[:, i]
                W_new[:, idx] = W[:, i]
                idx += 1
        
        # Add small noise to the last column
        U_new[:, -1] += np.random.randn(n * m) * 0.1
        V_new[:, -1] += np.random.randn(m * p) * 0.1
        W_new[:, -1] += np.random.randn(p * n) * 0.1
        
        # Run ALS
        for it in range(100):
            if time.time() > deadline:
                break
            U_new = _als_update_U(T, V_new, W_new)
            V_new = _als_update_V(T, U_new, W_new)
            W_new = _als_update_W(T, U_new, V_new)
        
        if _check_decomposition(U_new, V_new, W_new, T, tol=1e-7):
            polished = _polish_rational(U_new, V_new, W_new, T)
            if polished is not None:
                return polished
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None
=======
def _try_rank17_als(T, deadline):
    """
    Try to find a rank-17 decomposition using tensor unfolding analysis and ALS.
    Uses the mode-1, mode-2, mode-3 unfoldings to guide the search.
    """
    n, m, p = DIMENSIONS
    target_rank = 17
    nU, nV, nW = n*m, m*p, p*n  # 4, 10, 10
    
    U_struct, V_struct, W_struct = _build_structured_decomposition()
    
    # Analyze the tensor unfoldings to understand the structure
    # T is (4, 10, 10) with 100 non-zero entries
    T_flat = T.reshape(nU, -1)  # 4 x 100
    T_flat_v = T.reshape(nV, -1).T  # 10 x 100 -> 100 x 10, transpose to 10x100
    T_flat_w = T.reshape(nW, -1).T  # 10 x 100 -> 100 x 10, transpose to 10x100
    
    # For rank r, we need:
    # U (4xr) * (V*W outer product reshaped) = T_unfold1
    # The key insight: if we fix V and W, U is determined by linear system
    # Similarly for V and W
    
    # Strategy: Use the fact that the tensor has a specific sparsity pattern
    # Try to find a rank-17 solution by exploiting the structure
    
    # First, try a direct linear algebra approach:
    # If we assume the last 4 terms handle column 4 (schoolbook),
    # we need rank 13 for the first 4 columns (2x2x4 part)
    # 2x2x4 can be done with 2 Strassen blocks = rank 14, so we need to save 1
    
    # Alternative: try to merge one term across blocks
    # Let's try a more aggressive ALS with better initialization
    
    for seed in range(30):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize from structured solution with perturbation
        U_new = U_struct.copy()
        V_new = V_struct.copy()
        W_new = W_struct.copy()
        
        # Remove one column
        remove_col = seed % 18
        
        U_r = np.zeros((nU, target_rank), dtype=np.float64)
        V_r = np.zeros((nV, target_rank), dtype=np.float64)
        W_r = np.zeros((nW, target_rank), dtype=np.float64)
        
        idx = 0
        for i in range(18):
            if i != remove_col:
                U_r[:, idx] = U_struct[:, i]
                V_r[:, idx] = V_struct[:, i]
                W_r[:, idx] = W_struct[:, i]
                idx += 1
        
        # Add structured noise based on the tensor structure
        noise_scale = 0.05 + 0.01 * (seed % 5)
        U_r[:, -1] += np.random.randn(nU) * noise_scale
        V_r[:, -1] += np.random.randn(nV) * noise_scale
        W_r[:, -1] += np.random.randn(nW) * noise_scale
        
        # Run ALS with early stopping
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for it in range(200):
            if time.time() > deadline:
                break
            
            U_r = _als_update_U(T, V_r, W_r)
            V_r = _als_update_V(T, U_r, W_r)
            W_r = _als_update_W(T, U_r, V_r)
            
            # Check convergence
            err = np.linalg.norm(_decompose(U_r, V_r, W_r) - T, 'fro')
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
            
            if err < 1e-10:
                break
        
        if best_err < 1e-7:
            polished = _polish_rational(best_U, best_V, best_W, T)
            if polished is not None:
                return polished
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None
>>>>>>> REPLACE

