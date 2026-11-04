    def rank_reduction(U, V, W, max_time=100.0):
        """
        Rank reduction via tensor matrix linear dependence.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        start_opt = time.time()
        tol = 1e-10
        
        n, m, p = DIMENSIONS
        num_entries = n * m * p
        
        # Pre-compute indices for tensor matrix
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        def build_tensor_matrix(U, V, W):
            # U shape (n*m, rank), V (m*p, rank), W (p*n, rank)
            U_sel = U[i_idx * m + j_idx, :]
            V_sel = V[j_idx * p + k_idx, :]
            W_sel = W[k_idx * n + i_idx, :]
            return U_sel * V_sel * W_sel
        
        # Initial tensor matrix
        M = build_tensor_matrix(U, V, W)
        
        for iteration in range(200):
            if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                break
            
            reduced = False
            # Check columns from last to first
            for r in range(rank - 1, -1, -1):
                if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                    break
                if rank <= 1:
                    break
                
                # Check if column r is in span of others
                # Use SVD or QR to check rank
                # M_other = M[:, [i for i in range(rank) if i != r]]
                # M_col = M[:, r]
                # Check if rank(M) == rank(M_other)
                
                # Efficient check:
                # Compute rank of M and M without column r
                # Since M is 84x78, rank is at most 78.
                # We can use np.linalg.matrix_rank
                
                try:
                    # Remove column r
                    mask = [i for i in range(rank) if i != r]
                    M_other = M[:, mask]
                    
                    # Check rank
                    # Using SVD might be slow if done repeatedly.
                    # Alternative: solve M_other * x = M_col
                    # If residual is small, it's dependent.
                    x, residuals, rank_val, s = np.linalg.lstsq(M_other, M[:, r], rcond=tol)
                    if len(residuals) > 0 and residuals[0] < tol * 10:
                        # Dependent
                        U = np.delete(U, r, axis=1)
                        V = np.delete(V, r, axis=1)
                        W = np.delete(W, r, axis=1)
                        M = np.delete(M, r, axis=1)
                        rank -= 1
                        reduced = True
                        break
                except Exception:
                    continue
            
            if not reduced:
                break
        
        return U, V, W, rank
