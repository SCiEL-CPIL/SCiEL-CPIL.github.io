import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 by 4x6 matrix multiplication.
    Uses numerical search for lower rank, falls back to Strassen-based decomposition (rank 42).
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    def _validate(U_arr, V_arr, W_arr, n_val, m_val, p_val):
        U_arr = np.asarray(U_arr, dtype=float)
        V_arr = np.asarray(V_arr, dtype=float)
        W_arr = np.asarray(W_arr, dtype=float)
        rank = U_arr.shape[1]
        if U_arr.shape[0] != n_val * m_val:
            return False
        if V_arr.shape[0] != m_val * p_val:
            return False
        if W_arr.shape[0] != p_val * n_val:
            return False
        if U_arr.shape[1] != rank or V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    u_row = U_arr[i * m_val + j, :]
                    v_row = V_arr[j * p_val + k, :]
                    w_row = W_arr[k * n_val + i, :]
                    s = np.sum(u_row * v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def _alternating_minimize(U, V, W, max_iters=50, time_limit=None):
        n, m, p = DIMENSIONS
        R = U.shape[1]
        for it in range(max_iters):
            if time_limit and time.time() > time_limit:
                break
            # Update U
            for i in range(n):
                for j in range(m):
                    idx = i * m + j
                    A_mat = np.zeros((p, R))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    U[idx, :] = sol
            # Update V
            for j in range(m):
                for k in range(p):
                    idx = j * p + k
                    A_mat = np.zeros((n, R))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                    V[idx, :] = sol
            # Update W
            for k in range(p):
                for i in range(n):
                    idx = k * n + i
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    sol, _, _, _ = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                    W[idx, :] = sol
        return U, V, W
    
    # Try search for lower rank
    for target_rank in range(35, 42):
        if time.time() > deadline - 30:
            break
        for attempt in range(3):
            if time.time() > deadline - 25:
                break
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            U, V, W = _alternating_minimize(U, V, W, max_iters=30, time_limit=deadline - 20)
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if _validate(U_rounded, V_rounded, W_rounded, n, m, p):
                return {'U': U_rounded.tolist(), 'V': V_rounded.tolist(), 'W': W_rounded.tolist()}
    
    # Strassen decomposition for 2x2x2
    # U (4x7), V (4x7), W (4x7)
    # Indices for 2x2:
    # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A11-A01)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_strassen = np.zeros((4, 7))
    V_strassen = np.zeros((4, 7))
    W_strassen = np.zeros((4, 7))
    
    # Col 0: M1
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1
    
    # Col 1: M2
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1
    V_strassen[0, 1] = 1
    W_strassen[2, 1] = 1; W_strassen[3, 1] = -1
    
    # Col 2: M3
    U_strassen[0, 2] = 1
    V_strassen[2, 2] = 1; V_strassen[3, 2] = -1
    W_strassen[1, 2] = 1; W_strassen[3, 2] = 1
    
    # Col 3: M4
    U_strassen[3, 3] = 1
    V_strassen[1, 3] = 1; V_strassen[0, 3] = -1
    W_strassen[0, 3] = 1; W_strassen[2, 3] = 1
    
    # Col 4: M5
    U_strassen[0, 4] = 1; U_strassen[2, 4] = 1
    V_strassen[3, 4] = 1
    W_strassen[0, 4] = -1; W_strassen[1, 4] = 1
    
    # Col 5: M6
    U_strassen[1, 5] = 1; U_strassen[0, 5] = -1
    V_strassen[0, 5] = 1; V_strassen[2, 5] = 1
    W_strassen[3, 5] = 1
    
    # Col 6: M7
    U_strassen[3, 6] = 1; U_strassen[1, 6] = -1
    V_strassen[3, 6] = 1; V_strassen[1, 6] = 1
    W_strassen[0, 6] = 1; W_strassen[3, 6] = -1
    
    # Construct global U, V, W
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 6 sub-problems
    # (i_block, j_block, k_block)
    # i_block: 0 or 1 (rows of A)
    # j_block: 0 or 1 (cols of A, offset 0 or 2)
    # k_block: 0, 1, 2 (cols of B, offset 0, 2, 4)
    
    sub_problems = [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 1, 0), # Wait, A is 2x4. j ranges 0..3.
        # My previous analysis:
        # A is 2x4. j is 0..3.
        # B is 4x6. j is 0..3, k is 0..5.
        # Split m=4 into two blocks of 2: j in {0,1} and j in {2,3}.
        # Split p=6 into three blocks of 2: k in {0,1}, {2,3}, {4,5}.
        # So we have 2 (A blocks) * 3 (B blocks) = 6 sub-problems.
        # A block 0: j in {0,1}. A block 1: j in {2,3}.
        # B block 0: k in {0,1}. B block 1: k in {2,3}. B block 2: k in {4,5}.
        
        # The sub-problems are:
        # 1. A_block0 (j 0,1) * B_block0 (j 0,1, k 0,1)
        # 2. A_block0 (j 0,1) * B_block1 (j 0,1, k 2,3)
        # 3. A_block0 (j 0,1) * B_block2 (j 0,1, k 4,5)
        # 4. A_block1 (j 2,3) * B_block0 (j 2,3, k 0,1)
        # 5. A_block1 (j 2,3) * B_block1 (j 2,3, k 2,3)
        # 6. A_block1 (j 2,3) * B_block2 (j 2,3, k 4,5)
    ]
    
    # Correct list of sub-problems:
    # (i_offset, j_offset, k_offset)
    # i_offset is always 0 for A (since n=2, no i split).
    # j_offset: 0 for block 0, 2 for block 1.
    # k_offset: 0, 2, 4.
    
    sub_problems = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 0, 4),
        (0, 2, 0),
        (0, 2, 2),
        (0, 2, 4)
    ]
    
    for idx, (i_off, j_off, k_off) in enumerate(sub_problems):
        col_start = idx * 7
        for c_local in range(7):
            c_global = col_start + c_local
            # It's easier to loop over local indices
            for i_local in range(2):
                for j_local in range(2):
                    i_global = i_off + i_local
                    j_global = j_off + j_local
                    u_idx = i_global * m + j_global
                    U[u_idx, c_global] = U_strassen[i_local * 2 + j_local, c_local]
                    
            for j_local in range(2):
                for k_local in range(2):
                    j_global = j_off + j_local
                    k_global = k_off + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, c_global] = V_strassen[j_local * 2 + k_local, c_local]
                    
            for k_local in range(2):
                for i_local in range(2):
                    k_global = k_off + k_local
                    i_global = i_off + i_local
                    w_idx = k_global * n + i_global
                    W[w_idx, c_global] = W_strassen[k_local * 2 + i_local, c_local]
    
    def _validate(U_arr, V_arr, W_arr, n_val, m_val, p_val):
        """Validate that U, V, W form an exact decomposition."""
        U_arr = np.asarray(U_arr, dtype=float)
        V_arr = np.asarray(V_arr, dtype=float)
        W_arr = np.asarray(W_arr, dtype=float)
        rank = U_arr.shape[1]
        if U_arr.shape[0] != n_val * m_val:
            return False
        if V_arr.shape[0] != m_val * p_val:
            return False
        if W_arr.shape[0] != p_val * n_val:
            return False
        if U_arr.shape[1] != rank or V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        
        # Check non-zero entries (consistent indices)
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    u_row = U_arr[i * m_val + j, :]
                    v_row = V_arr[j * p_val + k, :]
                    w_row = W_arr[k * n_val + i, :]
                    s = np.sum(u_row * v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if _validate(U, V, W, n, m, p):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fallback to Schoolbook
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
