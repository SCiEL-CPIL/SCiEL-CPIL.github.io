import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    n, m, p = 3, 4, 4

    # Build expected tensor T[a,b,c] where a=row*n_cols? No:
    # A is n x m = 3x4, B is m x p = 4x4, C is n x p = 3x4
    # Tensor index: a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # A index: a_row = a // m, a_col = a % m
    # B index: b_row = b // p, b_col = b % p
    # C index: c follows W[k*n+i, r] convention where output C[i,j] has i in [0,n), j in [0,p)
    # c = j * n + i  (column-major for C)
    # T[a, b, c] = 1 if a_col == b_row and c = (b_col)*n + (a_row)
    
    T_exp = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        a_r, a_c = divmod(a, m)
        for b in range(m*p):
            b_r, b_c = divmod(b, p)
            if a_c == b_r:
                c_idx = b_c * n + a_r
                T_exp[a, b, c_idx] = 1.0

    # Block-based construction:
    # Split A's columns into groups. A is 3x4. We use 2x2 Strassen blocks.
    # Strategy: split A into row groups of 2, column groups of 2.
    # But A has 3 rows, so we handle rows 0,1 with Strassen and row 2 trivially.
    # Similarly for B: 4x4, split into 2x2 blocks.
    
    # Let's build a 3x2x2 block decomposition:
    # A_block: 3x2, B_block: 2x2, C_block: 3x2
    # Strassen for first 2 rows of A_block (2x2x2): rank 7
    # Trivial for row 2 of A_block (1x2x2): rank 4
    # Total per block: 11
    
    # Then tile: A has 4 columns = 2 blocks of 2 columns
    # B has 4 rows = 2 blocks of 2 rows, 4 columns = 2 blocks of 2 columns
    # So 2x2 = 4 blocks, each rank 11, total rank 44
    
    U_loc = np.zeros((6, 11))  # 3*2
    V_loc = np.zeros((4, 11))  # 2*2
    W_loc = np.zeros((6, 11))  # 3*2
    
    # Strassen for 2x2x2 (rows 0,1 of A; all of B; rows 0,1 of C)
    # S1: (A00+A11)(B00+B11) -> C00 += S1, C11 += S1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1
    
    # S2: (A10+A11)B00 -> C10 += S2, C11 -= S2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1
    
    # S3: A00(B10-B00) -> C01 += S3, C11 += S3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[0, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1
    
    # S4: A11(B01-B11) -> C00 += S4, C10 += S4
    U_loc[3, 3] = 1
    V_loc[1, 3] = 1; V_loc[3, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1
    
    # S5: (A00+A01)B11 -> C00 -= S5, C01 += S5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1
    
    # S6: (A10-A00)(B00+B10) -> C10 += S6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[2, 5] = 1
    
    # S7: (A01-A11)(B01+B11) -> C01 += S7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[1, 6] = 1; V_loc[3, 6] = 1
    W_loc[1, 6] = 1
    
    # Trivial for row 2 of A_block (1x2x2): rank 4
    U_loc[4, 7] = 1; V_loc[0, 7] = 1; W_loc[4, 7] = 1
    U_loc[5, 8] = 1; V_loc[2, 8] = 1; W_loc[4, 8] = 1
    U_loc[4, 9] = 1; V_loc[1, 9] = 1; W_loc[5, 9] = 1
    U_loc[5, 10] = 1; V_loc[3, 10] = 1; W_loc[5, 10] = 1
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_col_start):
        """Add a 3x2x2 block at the given positions"""
        col_offset = block_idx * 11
        
        # A: 3x2 block at (a_row_start, a_col_start)
        for r in range(3):
            for c in range(2):
                local_a_idx = r * 2 + c
                global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
        
        # B: 2x2 block at (b_row_start, b_col_start)
        for r in range(2):
            for c in range(2):
                local_b_idx = r * 2 + c
                global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
        
        # C: 3x2 block at (c_col_start), all rows
        for r in range(3):
            for c in range(2):
                local_c_idx = r * 2 + c
                # C output: row r, col (c_col_start + c)
                # c_idx = col * n + row = (c_col_start + c) * n + r
                global_c_idx = (c_col_start + c) * n + r
                W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]
    
    # 4 blocks: (a_col_start, b_row_start, c_col_start)
    add_block(0, 0, 0, 0, 0, 0)   # A cols 0-1, B rows 0-1, C cols 0-1
    add_block(1, 0, 2, 0, 2, 2)   # A cols 2-3, B rows 0-1, C cols 2-3
    add_block(2, 0, 0, 2, 0, 0)   # A cols 0-1, B rows 2-3, C cols 0-1
    add_block(3, 0, 2, 2, 2, 2)   # A cols 2-3, B rows 2-3, C cols 2-3
    
    def verify(U_mat, V_mat, W_mat):
        """Verify decomposition is exact using einsum"""
        try:
            T = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
            return np.allclose(T, T_exp, atol=1e-9)
        except Exception:
            return False
    
    # Verify initial solution
    if not verify(U, V, W):
        # Fallback to trivial decomposition
        trivial_rank = n * m * p
        U_fallback = np.zeros((n*m, trivial_rank))
        V_fallback = np.zeros((m*p, trivial_rank))
        W_fallback = np.zeros((p*n, trivial_rank))
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
    
    # Try to reduce rank by zeroing out columns
    def try_reduce(U_cur, V_cur, W_cur, r_cur):
        removed = True
        while removed and time.time() < deadline:
            removed = False
            for col in range(r_cur - 1, -1, -1):
                if time.time() > deadline:
                    break
                U_test = np.delete(U_cur, col, axis=1)
                V_test = np.delete(V_cur, col, axis=1)
                W_test = np.delete(W_cur, col, axis=1)
                if verify(U_test, V_test, W_test):
                    U_cur, V_cur, W_cur = U_test, V_test, W_test
                    r_cur -= 1
                    removed = True
                    break
        return U_cur, V_cur, W_cur, r_cur
    
    U, V, W, rank = try_reduce(U, V, W, rank)
    
    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try pairwise column removal
    if time.time() < deadline - 10:
        r_cur = rank
        U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
        found_pair = True
        while found_pair and time.time() < deadline:
            found_pair = False
            for i in range(r_cur - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, r_cur):
                    if time.time() > deadline:
                        break
                    cols = list(set(range(r_cur)) - {i, j})
                    U_test = U_cur[:, cols]
                    V_test = V_cur[:, cols]
                    W_test = W_cur[:, cols]
                    if verify(U_test, V_test, W_test):
                        U_cur, V_cur, W_cur = U_test, V_test, W_test
                        r_cur -= 2
                        found_pair = True
                        break
                if found_pair:
                    break
        
        if r_cur < best_rank:
            best_rank = r_cur
            best_U, best_V, best_W = U_cur.copy(), V_cur.copy(), W_cur.copy()
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
