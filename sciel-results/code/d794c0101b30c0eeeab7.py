def _reduce_rank_via_tensor_nullspace(U, V, W, n, m, p, timeout=60.0):
    rank = U.shape[1]
    # Flatten each term tensor
    terms = []
    for r in range(rank):
        term = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term[i*m*p + j*p + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        terms.append(term)
    
    T_mat = np.array(terms)  # shape (rank, n*m*p)
    
    # Find null space of T_mat (rows that are linearly dependent)
    # Use SVD
    try:
        U_svd, s_vals, Vt = np.linalg.svd(T_mat, full_matrices=False)
        # Rank of T_mat
        tol = max(T_mat.shape) * np.max(s_vals) * 1e-10
        num_indep = np.sum(s_vals > tol)
        num_dependent = rank - num_indep
        
        if num_dependent == 0:
            return U, V, W, rank
        
        # The null space of T_mat^T gives coefficients for dependencies
        # Actually, we need null space of T_mat (as a map from R^rank to R^(nmp))
        # Wait, T_mat is (rank, nmp). We want to find vectors c in R^rank such that T_mat^T @ c = 0
        # That means sum_r c_r * term_r = 0
        
        # Null space of T_mat^T: vectors c where sum_r c_r * term_r = 0
        # T_mat^T is (nmp, rank), null space is in R^rank
        # Use SVD of T_mat: T_mat = U_svd @ diag(s) @ Vt
        # Null space of T_mat^T = last (rank - num_indep) columns of U_svd
        
        if num_indep < rank:
            null_vecs = U_svd[:, num_indep:]  # (rank, num_dependent)
            
            # Try to find integer/half-integer combinations
            # For each null vector, try to use it to eliminate terms
            # A null vector c means: sum_r c_r * term_r = 0
            # If c has a zero entry, we can't directly eliminate
            # If all entries are nonzero, we can express one term as combo of others
            
            # Strategy: for each null vector, find the entry with smallest |c_r|
            # and try to eliminate that term
            best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
            
            for nv in null_vecs:
                # Find index with smallest absolute value (candidate to eliminate)
                abs_nv = np.abs(nv)
                min_idx = np.argmin(abs_nv)
                if abs_nv[min_idx] < 1e-10:
                    continue
                
                # Express term[min_idx] in terms of others
                # c[min_idx] * term[min_idx] = -sum_{r != min_idx} c[r] * term[r]
                # term[min_idx] = -sum_{r != min_idx} (c[r]/c[min_idx]) * term[r]
                
                # But this doesn't directly help us reduce rank unless we can
                # absorb the coefficients into U, V, W
                # This is tricky because the coefficients are mixed across all three matrices
                
                pass
            
            # Alternative: try greedy elimination using null vectors
            # Pick a null vector, identify which term to remove, and adjust others
            # This is complex; let's try a simpler approach first
            
            # Try: for each column, check if it's in the span of others
            # by checking if removing it still gives the correct tensor
            current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
            current_rank = rank
            
            for _ in range(20):
                if time.time() - start > timeout:
                    break
                found = False
                for col in range(current_rank):
                    if time.time() - start > timeout:
                        break
                    mask = np.ones(current_rank, dtype=bool)
                    mask[col] = False
                    U_test = current_U[:, mask]
                    V_test = current_V[:, mask]
                    W_test = current_W[:, mask]
                    if _verify(U_test, V_test, W_test, n, m, p):
                        current_U, current_V, current_W = U_test, V_test, W_test
                        current_rank = U_test.shape[1]
                        found = True
                        break
                if not found:
                    break
            
            return current_U, current_V, current_W, current_rank
    except Exception:
        pass
    
    return U, V, W, rank
