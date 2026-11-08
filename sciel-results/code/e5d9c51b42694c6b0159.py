"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    import numpy as np
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # 2x2x2 Strassen rank-7 decomposition
    S_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=float)
    S_V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=float)
    S_W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=float)

    n_blocks = n // 2  # 2
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 3

    total_rank = n_blocks * m_blocks * p_blocks * 7  # 84

    # Build U, V, W matrices
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j, col] = S_U[di * 2 + dj, c]
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k, col] = S_V[dj * 2 + dk, c]
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i, col] = S_W[dk * 2 + di, c]
                col_offset += 7

    # Vectorized verification
    def verify_decomp(U_mat, V_mat, W_mat):
        """Check if decomposition is valid using vectorized computation."""
        rank = U_mat.shape[1]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k
        
        # Build index arrays for efficient computation
        # For each r, compute outer product contribution
        T = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u_col = U_mat[:, r]  # shape (n*m,)
            v_col = V_mat[:, r]  # shape (m*p,)
            w_col = W_mat[:, r]  # shape (p*n,)
            
            # Reshape u_col to (n, m), v_col to (m, p), w_col to (p, n)
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            
            # T[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_reshaped[i, j] * v_reshaped[j, k] * w_reshaped[k, i]
        
        return np.allclose(T, 1.0, atol=1e-9)

    # Faster verification using einsum-like approach
    def verify_decomp_fast(U_mat, V_mat, W_mat):
        """Faster verification using matrix operations."""
        rank = U_mat.shape[1]
        T = np.zeros((n, m, p), dtype=float)
        
        for r in range(rank):
            u_col = U_mat[:, r].reshape(n, m)
            v_col = V_mat[:, r].reshape(m, p)
            w_col = W_mat[:, r].reshape(p, n)
            
            # Compute sum over r: U[i,j] * V[j,k] * W[k,i]
            # For each (i,j): U[i,j] * (V[j,:] * W[:,i].T)
            for i in range(n):
                for j in range(m):
                    if u_col[i, j] != 0:
                        # V[j,:] has shape (p,), W[:,i] has shape (p,)
                        # We need sum_k V[j,k] * W[k,i]
                        val = np.dot(v_col[j, :], w_col[:, i])
                        T[i, j, :] += u_col[i, j] * val
        
        return np.allclose(T, 1.0, atol=1e-9)

    # Even faster: precompute for each column
    def compute_tensor_for_columns(U_mat, V_mat, W_mat, cols):
        """Compute tensor contribution from specified columns."""
        T = np.zeros((n, m, p), dtype=float)
        for r in cols:
            u_col = U_mat[:, r].reshape(n, m)
            v_col = V_mat[:, r].reshape(m, p)
            w_col = W_mat[:, r].reshape(p, n)
            
            for i in range(n):
                for j in range(m):
                    if u_col[i, j] != 0:
                        val = np.dot(v_col[j, :], w_col[:, i])
                        T[i, j, :] += u_col[i, j] * val
        return T

    # Verify initial decomposition
    if not verify_decomp_fast(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank
    # Strategy: compute the tensor for each column, then find linear dependencies
    
    # Precompute tensor contribution for each column
    tensor_cols = []
    for r in range(total_rank):
        t_col = compute_tensor_for_columns(U, V, W, [r])
        tensor_cols.append(t_col.flatten())
    
    tensor_matrix = np.array(tensor_cols, dtype=float)  # shape (total_rank, n*m*p)
    
    # Find the effective rank using SVD
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(tensor_matrix, full_matrices=False)
        tol = 1e-10
        effective_rank = int(np.sum(s_vals > tol))
        
        if effective_rank < total_rank:
            # Greedy elimination: try to remove columns that are linearly dependent
            # on the others
            
            # Sort columns by their singular value (smallest first - most likely redundant)
            col_order = np.argsort(s_vals)
            
            current_U = U.copy()
            current_V = V.copy()
            current_W = W.copy()
            current_rank = total_rank
            removed = set()
            
            # Try removing columns in order of smallest singular value
            max_checks = 50
            checks_done = 0
            
            for r in col_order:
                if checks_done >= max_checks:
                    break
                if time.time() > deadline - 20:
                    break
                if r in removed:
                    continue
                
                # Try removing column r
                keep_mask = [i for i in range(current_rank) if i not in removed]
                # Remove r from consideration
                if r in keep_mask:
                    new_keep = [i for i in keep_mask if i != r]
                    
                    if len(new_keep) == 0:
                        break
                    
                    new_U = current_U[:, new_keep]
                    new_V = current_V[:, new_keep]
                    new_W = current_W[:, new_keep]
                    
                    if verify_decomp_fast(new_U, new_V, new_W):
                        current_U = new_U
                        current_V = new_V
                        current_W = new_W
                        current_rank = len(new_keep)
                        removed.add(r)
                checks_done += 1
            
            if verify_decomp_fast(current_U, current_V, current_W):
                return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
                
    except Exception as e:
        pass

    # Fallback: try a different elimination strategy
    # Use Gaussian elimination on the tensor matrix to find a basis
    try:
        if time.time() < deadline - 30:
            # Find a maximal linearly independent subset
            # Use QR decomposition with column pivoting
            from numpy.linalg import qr
            
            # QR with column pivoting
            Q, R, P = qr(tensor_matrix.T, mode='complete')
            # P gives the column permutation
            # The diagonal of R indicates rank
            diag = np.abs(np.diag(R))
            tol = 1e-10
            rank_est = int(np.sum(diag > tol))
            
            if rank_est < total_rank:
                # The first rank_est columns (after permutation) form a basis
                # But we need to be careful - let's use the pivot information
                
                # Actually, let's use a simpler approach: iteratively check independence
                basis_cols = []
                basis_tensors = []
                
                for r in range(total_rank):
                    if time.time() > deadline - 25:
                        break
                    
                    # Check if tensor_cols[r] is in span of basis_tensors
                    if len(basis_tensors) == 0:
                        basis_cols.append(r)
                        basis_tensors.append(tensor_cols[r])
                    else:
                        basis_mat = np.array(basis_tensors)
                        # Check if tensor_cols[r] is in the span
                        # Solve basis_mat.T @ x = tensor_cols[r]
                        try:
                            # Use least squares
                            x, residuals, rank, sv = np.linalg.lstsq(basis_mat.T, tensor_cols[r], rcond=1e-10)
                            if residuals is None or len(residuals) == 0:
                                residuals = [np.linalg.norm(basis_mat.T @ x - tensor_cols[r])]
                            if residuals[0] < 1e-8:
                                # Linearly dependent, skip
                                continue
                        except:
                            pass
                        basis_cols.append(r)
                        basis_tensors.append(tensor_cols[r])
                
                if len(basis_cols) < current_rank:
                    new_U = U[:, basis_cols]
                    new_V = V[:, basis_cols]
                    new_W = W[:, basis_cols]
                    
                    if verify_decomp_fast(new_U, new_V, new_W):
                        return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
                        
    except Exception as e:
        pass

    # Final fallback
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
