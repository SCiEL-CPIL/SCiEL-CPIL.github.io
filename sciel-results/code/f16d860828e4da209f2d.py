"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    import time
    import numpy as np
    start_time = time.time()
    deadline = start_time + 145

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

    # Fast verification using einsum
    def verify_decomp(U_mat, V_mat, W_mat):
        """Check if decomposition is valid."""
        rank = U_mat.shape[1]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Need T[i,j,k] = 1 for all i,j,k
        
        T = np.zeros((n, m, p), dtype=float)
        for r in range(rank):
            u_col = U_mat[:, r].reshape(n, m)
            v_col = V_mat[:, r].reshape(m, p)
            w_col = W_mat[:, r].reshape(p, n)
            # T[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
            for i in range(n):
                for j in range(m):
                    if u_col[i, j] != 0:
                        T[i, j, :] += u_col[i, j] * (v_col[j, :] * w_col[:, i])
        
        return np.allclose(T, 1.0, atol=1e-9)

    # Compute tensor contribution for a set of columns
    def compute_tensor(U_mat, V_mat, W_mat, cols):
        T = np.zeros((n, m, p), dtype=float)
        for r in cols:
            u_col = U_mat[:, r].reshape(n, m)
            v_col = V_mat[:, r].reshape(m, p)
            w_col = W_mat[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    if u_col[i, j] != 0:
                        T[i, j, :] += u_col[i, j] * (v_col[j, :] * w_col[:, i])
        return T.flatten()

    # Verify initial decomposition
    if not verify_decomp(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strategy 1: SVD-based rank reduction
    try:
        if time.time() < deadline - 30:
            # Compute tensor for each column
            tensor_cols = []
            for r in range(total_rank):
                t_col = compute_tensor(U, V, W, [r])
                tensor_cols.append(t_col)
            
            tensor_matrix = np.array(tensor_cols, dtype=float).T  # shape (n*m*p, total_rank)
            
            # SVD to find effective rank
            U_svd, s_vals, Vt_svd = np.linalg.svd(tensor_matrix, full_matrices=False)
            tol = 1e-10
            effective_rank = int(np.sum(s_vals > tol))
            
            if effective_rank < total_rank:
                # Greedy elimination based on singular values
                col_order = np.argsort(s_vals)
                
                current_U = U.copy()
                current_V = V.copy()
                current_W = W.copy()
                removed = set()
                
                max_checks = 100
                checks_done = 0
                
                for r in col_order:
                    if checks_done >= max_checks:
                        break
                    if time.time() > deadline - 20:
                        break
                    if r in removed:
                        continue
                    
                    new_keep = [i for i in range(total_rank) if i != r and i not in removed]
                    new_U = current_U[:, new_keep]
                    new_V = current_V[:, new_keep]
                    new_W = current_W[:, new_keep]
                    
                    if verify_decomp(new_U, new_V, new_W):
                        current_U = new_U
                        current_V = new_V
                        current_W = new_W
                        removed.add(r)
                    checks_done += 1
                
                if verify_decomp(current_U, current_V, current_W):
                    best_rank = current_U.shape[1]
                    best_U = current_U.copy()
                    best_V = current_V.copy()
                    best_W = current_W.copy()
    except Exception:
        best_rank = total_rank
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()

    # Strategy 2: Randomized greedy elimination
    try:
        if time.time() < deadline - 15 and best_rank == total_rank:
            np.random.seed(42)
            current_U = U.copy()
            current_V = V.copy()
            current_W = W.copy()
            removed = set()
            
            max_checks = 50
            checks_done = 0
            
            while checks_done < max_checks:
                if time.time() > deadline - 15:
                    break
                
                # Randomly pick a column to try removing
                available = [i for i in range(total_rank) if i not in removed]
                if len(available) <= 1:
                    break
                
                r = np.random.choice(available)
                new_keep = [i for i in range(total_rank) if i != r and i not in removed]
                
                new_U = current_U[:, new_keep]
                new_V = current_V[:, new_keep]
                new_W = current_W[:, new_keep]
                
                if verify_decomp(new_U, new_V, new_W):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    removed.add(r)
                    if current_U.shape[1] < best_rank:
                        best_rank = current_U.shape[1]
                        best_U = current_U.copy()
                        best_V = current_V.copy()
                        best_W = current_W.copy()
                checks_done += 1
    except Exception:
        pass

    # Return best result
    if best_rank < total_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

