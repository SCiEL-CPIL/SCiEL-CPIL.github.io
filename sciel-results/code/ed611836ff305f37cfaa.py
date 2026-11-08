"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    import numpy as np
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s limit

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

    # Build initial U, V, W matrices
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

    # Verify the decomposition
    def verify_decomp(U_mat, V_mat, W_mat):
        """Check if decomposition is valid: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1"""
        rank = U_mat.shape[1]
        T_decomp = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u_col = U_mat[:, r]
            v_col = V_mat[:, r]
            w_col = W_mat[:, r]
            # For each (i,j,k): u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
            # Vectorize over j for fixed i,k
            for i in range(n):
                for k in range(p):
                    # u_col[i*m+j] for j in range(m)
                    u_vals = u_col[i*m:(i+1)*m]  # shape (m,)
                    # v_col[j*p+k] for j in range(m)
                    v_vals = v_col[k::p][:m]  # shape (m,)
                    # w_col[k*n+i] is scalar
                    w_val = w_col[k*n+i]
                    T_decomp[i, :, k] += u_vals * v_vals * w_val
        return np.allclose(T_decomp, 1.0, atol=1e-9)

    # Verify current decomposition
    if not verify_decomp(U, V, W):
        # Shouldn't happen, but return as-is
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by removing redundant columns
    # A column r is redundant if the tensor can still be represented without it
    # i.e., there exist coefficients such that col_r = sum_{r' != r} c_{r'} col_{r'}
    # in the tensor product space
    
    # More efficiently: check if removing column r still gives a valid decomposition
    # This is O(rank) checks, each O(n*m*p*rank) - too slow for 84 columns
    
    # Alternative approach: use the structure to find linear dependencies
    # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r where u_r, v_r, w_r are the r-th columns
    # If we can find a linear combination of columns that gives zero tensor, we can remove one
    
    # Let's try a greedy approach: for each column, check if it's linearly dependent
    # on the others in a way that preserves the tensor
    
    # Actually, let's try a simpler approach: randomly permute columns and try to remove them
    # one by one, checking validity each time
    
    # Better: use the fact that the Strassen decomposition has specific structure
    # The 7 terms in Strassen are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B11-B21)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For our 4x4x6 case, we have 2x2x3 blocks
    # Let's see if we can merge some blocks
    
    # Try removing columns one by one (greedy)
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = total_rank
    removed = [False] * total_rank
    
    # For efficiency, precompute the tensor contribution of each column
    # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def compute_tensor_col(r):
        """Compute the tensor contribution of column r"""
        u_col = current_U[:, r]
        v_col = current_V[:, r]
        w_col = current_W[:, r]
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for k in range(p):
                u_vals = u_col[i*m:(i+1)*m]
                v_vals = v_col[k::p][:m]
                w_val = w_col[k*n+i]
                T[i, :, k] += u_vals * v_vals * w_val
        return T

    # Precompute tensor contributions for all columns
    tensor_cols = [compute_tensor_col(r) for r in range(total_rank)]
    total_tensor = np.sum(tensor_cols, axis=0)
    
    # Now try to remove columns greedily
    # For column r, it's removable if tensor_cols[r] can be expressed as a linear combination
    # of the remaining tensor_cols
    # This is a linear algebra problem in R^{n*m*p}
    
    # For efficiency, let's try a simpler heuristic:
    # Check if removing column r still gives a valid decomposition
    # This is expensive, so let's limit the number of checks
    
    # Alternative: use SVD or rank checking on the tensor space
    # Flatten each tensor column to a vector of length n*m*p
    vec_dim = n * m * p
    tensor_matrix = np.zeros((total_rank, vec_dim), dtype=float)
    for r in range(total_rank):
        tensor_matrix[r] = tensor_cols[r].flatten()
    
    # Check rank of tensor_matrix
    # If rank < total_rank, there are linear dependencies
    # We can remove columns corresponding to the null space
    
    # Use SVD to find the rank
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(tensor_matrix, full_matrices=False)
        # Find the effective rank
        tol = 1e-10
        effective_rank = np.sum(s_vals > tol)
        
        if effective_rank < total_rank:
            # There are dependencies
            # The columns corresponding to small singular values are redundant
            # Remove columns in order of smallest singular value
            # But we need to be careful: removing one column changes the dependencies
            
            # Greedy: sort columns by their contribution to the null space
            # and try to remove them one by one
            
            # For now, let's just try removing columns with smallest singular values
            # and check if the decomposition is still valid
            
            # Get the null space basis
            null_space = Vt_svd[effective_rank:]  # shape (total_rank - effective_rank, total_rank)
            
            # Find columns that are most "in the null space"
            # A column r is redundant if there exists a vector c in the null space
            # such that c[r] != 0 and the tensor sum is preserved
            
            # Simpler approach: try removing each column and check validity
            # Limit the number of checks to stay within time budget
            
            max_checks = 20
            checks_done = 0
            removed_count = 0
            
            # Try to remove columns, starting with those most likely to be redundant
            # Sort by the magnitude of their projection onto the null space
            null_proj = np.zeros(total_rank, dtype=float)
            for r in range(total_rank):
                if effective_rank < total_rank:
                    null_proj[r] = np.linalg.norm(null_space[:, r])
            
            # Sort columns by null projection (descending) - most redundant first
            col_order = np.argsort(-null_proj)
            
            for r in col_order:
                if checks_done >= max_checks:
                    break
                if time.time() > deadline - 10:
                    break
                if removed[r]:
                    continue
                
                # Try removing column r
                new_rank = current_rank - 1
                new_U = np.delete(current_U, r, axis=1)
                new_V = np.delete(current_V, r, axis=1)
                new_W = np.delete(current_W, r, axis=1)
                
                # Check validity
                if verify_decomp(new_U, new_V, new_W):
                    # Successfully removed
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    current_rank = new_rank
                    removed[r] = True
                    removed_count += 1
                    # Update tensor matrix (remove row r)
                    tensor_matrix = np.delete(tensor_matrix, r, axis=0)
                    tensor_cols.pop(r)
                    # Reindex
                    for i in range(r, total_rank - removed_count):
                        if not removed[i]:
                            removed[i] = removed[i]  # keep as is
                            break
                    # Actually, let's just rebuild if needed
                    # For now, continue with the greedy approach
                checks_done += 1
            
            if removed_count > 0:
                # Rebuild the arrays without removed columns
                keep_indices = [r for r in range(total_rank) if not removed[r]]
                current_U = U[:, keep_indices]
                current_V = V[:, keep_indices]
                current_W = W[:, keep_indices]
                current_rank = len(keep_indices)
                
    except Exception as e:
        pass

    # Final verification
    if not verify_decomp(current_U, current_V, current_W):
        # Fallback to original
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
# EVOLVE-BLOCK-END
