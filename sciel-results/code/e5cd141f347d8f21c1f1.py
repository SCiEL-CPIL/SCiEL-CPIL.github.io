import time
import numpy as np

_DIM = (6, 6, 6)

def _residual_max(U, V, W, n, m, p):
    U_3d = U.reshape(n, m, -1)
    V_3d = V.reshape(m, p, -1)
    W_3d = W.reshape(p, n, -1)
    R_3d = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    residual = np.max(np.abs(R_3d - 1.0))
    return float(residual)

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Row indexing:
    U row = i*2 + j  (A_{i,j})
    V row = j*2 + k  (B_{j,k})
    W row = k*2 + i  (C_{k,i})
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1
    
    # M3 = A11(B21 - B22)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[0, 2] = 1
    
    # M4 = A22(B12 - B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _block_decomposition_4_2():
    """
    4+2 block Strassen decomposition for 6x6 matrix multiplication.
    Splits 6 into 4+2, giving 2x2 block structure.
    Uses Strassen on the block level where possible.
    """
    # Block sizes: n1=4, n2=2
    # A = [[A11(4x4), A12(4x2)], [A21(2x4), A22(2x2)]]
    # B = [[B11(4x4), B12(4x2)], [B21(2x4), B22(2x2)]]
    # C = [[C11(4x4), C12(4x2)], [C21(2x4), C22(2x2)]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # We can use Strassen-like formulas on the block level:
    # M1 = A11 * B11
    # M2 = A12 * B21
    # M3 = A21 * B11
    # M4 = A22 * B21
    # M5 = A11 * B12
    # M6 = A12 * B22
    # M7 = A21 * B12
    # M8 = A22 * B22
    #
    # But we can combine some using Strassen tricks on the 2x2 block structure.
    # Let's use a simpler approach: direct block multiplication with
    # Strassen for 4x4 and schoolbook for smaller blocks.
    
    n1, n2 = 4, 2
    
    # We'll build the decomposition by combining:
    # - Strassen for 4x4x4 (rank 49 = 7*7) for A11*B11
    # - Schoolbook for other blocks
    
    # Actually, let's use a cleaner approach: treat the 6x6 as a 2x2 block
    # matrix and apply Strassen at the block level, then decompose each
    # block multiplication.
    
    # Block-level Strassen for 2x2 blocks of size (4,2):
    # M1 = (A11+A22)(B11+B22) - but sizes don't match! A11 is 4x4, A22 is 2x2
    # This doesn't work directly.
    
    # Let's use a different approach: direct enumeration with optimization.
    # Total schoolbook rank for 6x6 is 216.
    # We can reduce by using Strassen for the 4x4 sub-problems.
    
    # Approach: decompose into 8 block multiplications:
    # 1. A11*B11: 4x4x4 -> use Strassen rank 49 (vs schoolbook 64)
    # 2. A12*B21: 4x2x2 -> schoolbook rank 16
    # 3. A21*B11: 2x4x4 -> schoolbook rank 32
    # 4. A22*B21: 2x2x2 -> Strassen rank 7 (vs schoolbook 8)
    # 5. A11*B12: 4x4x2 -> schoolbook rank 32
    # 6. A12*B22: 4x2x2 -> schoolbook rank 16
    # 7. A21*B12: 2x4x2 -> schoolbook rank 16
    # 8. A22*B22: 2x2x2 -> Strassen rank 7 (vs schoolbook 8)
    
    # Total: 49 + 16 + 32 + 7 + 32 + 16 + 16 + 7 = 175
    
    # Let's implement this.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to get row indices for a block
    def get_block_rows(n_start, m_start, n_size, m_size, n_total, m_total):
        """Get row indices for U (A matrix) block at position (n_start, m_start)"""
        rows = []
        for i in range(n_start, n_start + n_size):
            for j in range(m_start, m_start + m_size):
                rows.append(i * m_total + j)
        return rows
    
    def get_block_rows_V(m_start, p_start, m_size, p_size, m_total, p_total):
        """Get row indices for V (B matrix) block"""
        rows = []
        for j in range(m_start, m_start + m_size):
            for k in range(p_start, p_start + p_size):
                rows.append(j * p_total + k)
        return rows
    
    def get_block_rows_W(p_start, n_start, p_size, n_size, p_total, n_total):
        """Get row indices for W (C matrix) block"""
        rows = []
        for k in range(p_start, p_start + p_size):
            for i in range(n_start, n_start + n_size):
                rows.append(k * n_total + i)
        return rows
    
    n_total, m_total, p_total = 6, 6, 6
    
    # Block 1: A11(4x4) * B11(4x4) -> C11 contribution
    # Use Strassen for 4x4x4
    U4, V4, W4 = _strassen_4x4()
    for r in range(49):
        u_vec = np.zeros(36)
        v_vec = np.zeros(36)
        w_vec = np.zeros(36)
        # Map 4x4 Strassen indices to 6x6 positions
        # A11 occupies rows 0-3, cols 0-3 in A
        # B11 occupies rows 0-3, cols 0-3 in B
        # C11 occupies rows 0-3, cols 0-3 in C
        for idx_u, val_u in enumerate(U4[:, r]):
            i_local, j_local = idx_u // 4, idx_u % 4
            u_vec[i_local * 6 + j_local] = val_u
        for idx_v, val_v in enumerate(V4[:, r]):
            j_local, k_local = idx_v // 4, idx_v % 4
            v_vec[j_local * 6 + k_local] = val_v
        for idx_w, val_w in enumerate(W4[:, r]):
            k_local, i_local = idx_w // 4, idx_w % 4
            w_vec[k_local * 6 + i_local] = val_w
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    # Block 2: A12(4x2) * B21(2x4) -> C11 contribution
    # A12: rows 0-3, cols 4-5
    # B21: rows 4-5, cols 0-3
    # C11: rows 0-3, cols 0-3
    # Schoolbook: 4*2*4 = 32 terms
    for i in range(4):
        for j in range(2):
            for k in range(4):
                u_vec = np.zeros(36)
                v_vec = np.zeros(36)
                w_vec = np.zeros(36)
                u_vec[i * 6 + (4 + j)] = 1.0  # A[i, 4+j]
                v_vec[(4 + j) * 6 + k] = 1.0  # B[4+j, k]
                w_vec[k * 6 + i] = 1.0  # C[k, i]
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
    
    # Block 3: A21(2x4) * B11(4x4) -> C21 contribution
    # A21: rows 4-5, cols 0-3
    # B11: rows 0-3, cols 0-3
    # C21: rows 4-5, cols 0-3
    # Schoolbook: 2*4*4 = 32 terms
    for i in range(2):
        for j in range(4):
            for k in range(4):
                u_vec = np.zeros(36)
                v_vec = np.zeros(36)
                w_vec = np.zeros(36)
                u_vec[(4 + i) * 6 + j] = 1.0  # A[4+i, j]
                v_vec[j * 6 + k] = 1.0  # B[j, k]
                w_vec[k * 6 + (4 + i)] = 1.0  # C[k, 4+i]
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
    
    # Block 4: A22(2x2) * B21(2x4) -> C21 contribution
    # A22: rows 4-5, cols 4-5
    # B21: rows 4-5, cols 0-3
    # C21: rows 4-5, cols 0-3
    # Use Strassen for 2x2x2 part, but B21 is 2x4
    # Actually, this is 2x2 * 2x4 -> 2x4
    # Schoolbook: 2*2*4 = 16 terms
    for i in range(2):
        for j in range(2):
            for k in range(4):
                u_vec = np.zeros(36)
                v_vec = np.zeros(36)
                w_vec = np.zeros(36)
                u_vec[(4 + i) * 6 + (4 + j)] = 1.0  # A[4+i, 4+j]
                v_vec[(4 + j) * 6 + k] = 1.0  # B[4+j, k]
                w_vec[k * 6 + (4 + i)] = 1.0  # C[k, 4+i]
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
    
    # Block 5: A11(4x4) * B12(4x2) -> C12 contribution
    # A11: rows 0-3, cols 0-3
    # B12: rows 0-3, cols 4-5
    # C12: rows 0-3, cols 4-5
    # Schoolbook: 4*4*2 = 32 terms
    for i in range(4):
        for j in range(4):
            for k in range(2):
                u_vec = np.zeros(36)
                v_vec = np.zeros(36)
                w_vec = np.zeros(36)
                u_vec[i * 6 + j] = 1.0  # A[i, j]
                v_vec[j * 6 + (4 + k)] = 1.0  # B[j, 4+k]
                w_vec[(4 + k) * 6 + i] = 1.0  # C[4+k, i]
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
    
    # Block 6: A12(4x2) * B22(2x2) -> C12 contribution
    # A12: rows 0-3, cols 4-5
    # B22: rows 4-5, cols 4-5
    # C12: rows 0-3, cols 4-5
    # Schoolbook: 4*2*2 = 16 terms
    for i in range(4):
        for j in range(2):
            for k in range(2):
                u_vec = np.zeros(36)
                v_vec = np.zeros(36)
                w_vec = np.zeros(36)
                u_vec[i * 6 + (4 + j)] = 1.0  # A[i, 4+j]
                v_vec[(4 + j) * 6 + (4 + k)] = 1.0  # B[4+j, 4+k]
                w_vec[(4 + k) * 6 + i] = 1.0  # C[4+k, i]
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
    
    # Block 7: A21(2x4) * B12(4x2) -> C22 contribution
    # A21: rows 4-5, cols 0-3
    # B12: rows 0-3, cols 4-5
    # C22: rows 4-5, cols 4-5
    # Schoolbook: 2*4*2 = 16 terms
    for i in range(2):
        for j in range(4):
            for k in range(2):
                u_vec = np.zeros(36)
                v_vec = np.zeros(36)
                w_vec = np.zeros(36)
                u_vec[(4 + i) * 6 + j] = 1.0  # A[4+i, j]
                v_vec[j * 6 + (4 + k)] = 1.0  # B[j, 4+k]
                w_vec[(4 + k) * 6 + (4 + i)] = 1.0  # C[4+k, 4+i]
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
    
    # Block 8: A22(2x2) * B22(2x2) -> C22 contribution
    # A22: rows 4-5, cols 4-5
    # B22: rows 4-5, cols 4-5
    # C22: rows 4-5, cols 4-5
    # Use Strassen for 2x2x2
    U2, V2, W2 = _strassen_2x2()
    for r in range(7):
        u_vec = np.zeros(36)
        v_vec = np.zeros(36)
        w_vec = np.zeros(36)
        for idx_u, val_u in enumerate(U2[:, r]):
            i_local, j_local = idx_u // 2, idx_u % 2
            u_vec[(4 + i_local) * 6 + (4 + j_local)] = val_u
        for idx_v, val_v in enumerate(V2[:, r]):
            j_local, k_local = idx_v // 2, idx_v % 2
            v_vec[(4 + j_local) * 6 + (4 + k_local)] = val_v
        for idx_w, val_w in enumerate(W2[:, r]):
            k_local, i_local = idx_w // 2, idx_w % 2
            w_vec[(4 + k_local) * 6 + (4 + i_local)] = val_w
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
    
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    return U, V, W

def _strassen_4x4():
    """
    Returns U, V, W for 4x4 matrix multiplication with rank 49 (7*7).
    Uses tensor product of two 2x2 Strassen decompositions.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U2, V2, W2 = _strassen_2x2()
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # 4x4 = 2x2 blocks of 2x2
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # Row index for U: (i1*2+i2)*4 + (j1*2+j2) = 8*i1 + 4*i2 + 2*j1 + j2
    # Col index: r1*7 + r2 where r1,r2 in 0..6
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    row_u = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    
                    for r1 in range(7):
                        u2_val = U2[i1 * 2 + j1, r1]
                        if u2_val == 0: continue
                        
                        for r2 in range(7):
                            u3_val = U2[i2 * 2 + j2, r2]
                            if u3_val == 0: continue
                            
                            col = r1 * 7 + r2
                            U[row_u, col] = u2_val * u3_val
    
    # V: row = (j1*2+j2)*4 + (k1*2+k2) = 8*j1 + 4*j2 + 2*k1 + k2
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    row_v = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                    
                    for r1 in range(7):
                        v2_val = V2[j1 * 2 + k1, r1]
                        if v2_val == 0: continue
                        
                        for r2 in range(7):
                            v3_val = V2[j2 * 2 + k2, r2]
                            if v3_val == 0: continue
                            
                            col = r1 * 7 + r2
                            V[row_v, col] = v2_val * v3_val
    
    # W: row = (k1*2+k2)*4 + (i1*2+i2) = 8*k1 + 4*k2 + 2*i1 + i2
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    row_w = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                    
                    for r1 in range(7):
                        w2_val = W2[k1 * 2 + i1, r1]
                        if w2_val == 0: continue
                        
                        for r2 in range(7):
                            w3_val = W2[k2 * 2 + i2, r2]
                            if w3_val == 0: continue
                            
                            col = r1 * 7 + r2
                            W[row_w, col] = w2_val * w3_val
    
    return U, V, W

def construct():
    n, m, p = _DIM
    deadline = time.time() + 115.0  # Safety margin

    U_s, V_s, W_s = _schoolbook(n, m, p)
    best = (U_s.copy(), V_s.copy(), W_s.copy())
    best_rank = U_s.shape[1]
    
    # Try 4+2 block decomposition
    if time.time() < deadline:
        try:
            U_new, V_new, W_new = _block_decomposition_4_2()
            res = _residual_max(U_new, V_new, W_new, n, m, p)
            if res < 1e-7:
                rank_new = U_new.shape[1]
                if rank_new < best_rank:
                    best = (U_new.copy(), V_new.copy(), W_new.copy())
                    best_rank = rank_new
        except Exception:
            pass
    
    # Try original Strassen tensor product as fallback
    if time.time() < deadline and best_rank == 216:
        U2, V2, W2 = _strassen_2x2()
        U3, V3, W3 = _schoolbook(3, 3, 3)
        
        rank_strassen = 7 * 27  # 189
        U_new = np.zeros((36, rank_strassen))
        V_new = np.zeros((36, rank_strassen))
        W_new = np.zeros((36, rank_strassen))
        
        for i1 in range(2):
            for i2 in range(3):
                for j1 in range(2):
                    for j2 in range(3):
                        row_u = (3 * i1 + i2) * 6 + (3 * j1 + j2)
                        for r1 in range(7):
                            u2_val = U2[i1 * 2 + j1, r1]
                            if u2_val == 0: continue
                            for r2 in range(27):
                                u3_val = U3[i2 * 3 + j2, r2]
                                if u3_val == 0: continue
                                col = r1 * 27 + r2
                                U_new[row_u, col] = u2_val * u3_val
        
        for j1 in range(2):
            for j2 in range(3):
                for k1 in range(2):
                    for k2 in range(3):
                        row_v = (3 * j1 + j2) * 6 + (3 * k1 + k2)
                        for r1 in range(7):
                            v2_val = V2[j1 * 2 + k1, r1]
                            if v2_val == 0: continue
                            for r2 in range(27):
                                v3_val = V3[j2 * 3 + k2, r2]
                                if v3_val == 0: continue
                                col = r1 * 27 + r2
                                V_new[row_v, col] = v2_val * v3_val
        
        for k1 in range(2):
            for k2 in range(3):
                for i1 in range(2):
                    for i2 in range(3):
                        row_w = (3 * k1 + k2) * 6 + (3 * i1 + i2)
                        for r1 in range(7):
                            w2_val = W2[k1 * 2 + i1, r1]
                            if w2_val == 0: continue
                            for r2 in range(27):
                                w3_val = W3[k2 * 3 + i2, r2]
                                if w3_val == 0: continue
                                col = r1 * 27 + r2
                                W_new[row_w, col] = w2_val * w3_val
        
        res = _residual_max(U_new, V_new, W_new, n, m, p)
        if res < 1e-7:
            best = (U_new.copy(), V_new.copy(), W_new.copy())
            best_rank = rank_strassen

    Uf, Vf, Wf = best
    Uo = [[round(float(v), 9) for v in row] for row in Uf]
    Vo = [[round(float(v), 9) for v in row] for row in Vf]
    Wo = [[round(float(v), 9) for v in row] for row in Wf]
    return {"U": Uo, "V": Vo, "W": Wo}
