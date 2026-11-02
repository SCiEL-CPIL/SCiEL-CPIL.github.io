import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = 3, 4, 4
    rank = 44
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Local factors for 3x2x2 multiplication
    # A is 3x2, B is 2x2, C is 3x2
    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))
    
    # Strassen part (cols 0-6) for the 2x2x2 core (rows 0,1 of A)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1
    
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1
    
    # Extra part (cols 7-10) for Row 2 of A (trivial 1x2x2)
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        col_offset = block_idx * 11
        
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_row = a_row_start + r
                global_a_col = a_col_start + c
                global_a_idx = global_a_row * m + global_a_col
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
                
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_row = b_row_start + r
                global_b_col = b_col_start + c
                global_b_idx = global_b_row * p + global_b_col
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
                
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                global_c_row = c_row_start + r
                global_c_col = c_col_start + c
                global_c_idx = global_c_col * n + global_c_row
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

    add_block(0, 0, 0, 0, 0, 0, 0)
    add_block(1, 0, 2, 2, 0, 0, 0)
    add_block(2, 0, 0, 0, 2, 0, 2)
    add_block(3, 0, 2, 2, 2, 0, 2)
    
    # Build expected tensor
    T_exp = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        a_r, a_c = divmod(a, m)
        for b in range(m*p):
            b_r, b_c = divmod(b, p)
            if a_c == b_r:
                c_idx = b_c * n + a_r
                T_exp[a, b, c_idx] = 1.0
    
    def verify(U_mat, V_mat, W_mat, rnk):
        """Verify decomposition is exact"""
        try:
            T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
            return np.allclose(T, T_exp, atol=1e-9)
        except Exception:
            return False
    
    # Verify initial solution
    if not verify(U, V, W, rank):
        # Fallback to trivial
        U_fallback = np.zeros((n*m, n*m*p))
        V_fallback = np.zeros((m*p, n*m*p))
        W_fallback = np.zeros((p*n, n*m*p))
        for a in range(n*m):
            a_r, a_c = divmod(a, m)
            for b in range(m*p):
                b_r, b_c = divmod(b, p)
                if a_c == b_r:
                    c_idx = b_c * n + a_r
                    rank_idx = a * (m*p) + b
                    U_fallback[a, rank_idx] = 1
                    V_fallback[b, rank_idx] = 1
                    W_fallback[c_idx, rank_idx] = 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by zeroing out columns and checking
    # Strategy: try removing each column and see if it's still valid
    # Then try removing pairs, etc.
    
    def try_reduce(U_cur, V_cur, W_cur, r_cur):
        """Try to reduce rank by removing redundant columns"""
        if time.time() > deadline:
            return U_cur, V_cur, W_cur, r_cur
        
        removed = True
        while removed and time.time() < deadline:
            removed = False
            for col in range(r_cur - 1, -1, -1):
                if time.time() > deadline:
                    break
                # Try removing column col
                U_test = np.delete(U_cur, col, axis=1)
                V_test = np.delete(V_cur, col, axis=1)
                W_test = np.delete(W_cur, col, axis=1)
                
                if verify(U_test, V_test, W_test, r_cur - 1):
                    U_cur, V_cur, W_cur = U_test, V_test, W_test
                    r_cur -= 1
                    removed = True
                    break
        
        return U_cur, V_cur, W_cur, r_cur
    
    U, V, W, rank = try_reduce(U, V, W, rank)
    
    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try more aggressive: search for rank-1 term replacements
    # Use the fact that we can add/subtract rank-1 terms
    # Try to find if any combination of 2 rank-1 terms can replace 3, etc.
    
    # Alternative: use iterative deepening - try rank R from best_rank-1 downward
    # For each R, try to find a decomposition
    
    # For now, return the best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
