import numpy as np
import time

def construct():
    """
    Returns a rank-28 tensor decomposition for 2x4 x 4x4 matrix multiplication.
    Uses 4 independent Strassen 2x2x2 decompositions.
    """
    n, m, p = 2, 4, 4
    
    # Rank 28
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients for 2x2x2 multiplication
    # Indices for 2x2 matrix: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # U corresponds to A (2x2)
    # V corresponds to B (2x2)
    # W corresponds to C (2x2)
    
    # Standard Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define terms as (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # Using 0-based flat indexing for 2x2 matrices: 0,1,2,3
    
    strassen_defs = [
        # M1
        (
            [0, 3], [1, 1],      # U
            [0, 3], [1, 1],      # V
            [0, 3], [1, 1]       # W (C00, C11) -> Wait, C00 coeff 1, C11 coeff 1
        ),
        # M2
        (
            [2, 3], [1, 1],      # U
            [0], [1],            # V
            [2, 3], [1, -1]      # W (C10, C11)
        ),
        # M3
        (
            [0], [1],            # U
            [1, 3], [1, -1],     # V
            [1, 3], [1, 1]       # W (C01, C11)
        ),
        # M4
        (
            [3], [1],            # U
            [2, 0], [1, -1],     # V
            [0, 2], [1, 1]       # W (C00, C10)
        ),
        # M5
        (
            [0, 1], [1, 1],      # U
            [3], [1],            # V
            [0, 1], [-1, 1]      # W (C00, C01)
        ),
        # M6
        (
            [2, 0], [1, -1],     # U
            [0, 1], [1, 1],      # V
            [3], [1]             # W (C11)
        ),
        # M7
        (
            [1, 3], [1, -1],     # U
            [2, 3], [1, 1],      # V
            [0], [1]             # W (C00)
        ),
    ]
    
    # Blocks:
    # A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    # B is 4x4. Split into B_TL, B_TR, B_BL, B_BR (2x2 each).
    # C is 2x4. Split into C_L (cols 0,1) and C_R (cols 2,3).
    
    # Combinations:
    # 1. A_L * B_TL -> C_L
    # 2. A_R * B_BL -> C_L  (Note: A_R is cols 2,3 of A, B_BL is rows 2,3 of B)
    #    Wait, C_L = A_L B_TL + A_R B_BL.
    #    A_L: rows 0,1, cols 0,1.
    #    A_R: rows 0,1, cols 2,3.
    #    B_TL: rows 0,1, cols 0,1.
    #    B_BL: rows 2,3, cols 0,1.
    # 3. A_L * B_TR -> C_R
    # 4. A_R * B_BR -> C_R
    
    # Mapping functions for global indices
    # U index: r * 4 + c (r in 0..1, c in 0..3)
    # V index: r * 4 + c (r in 0..3, c in 0..3)
    # W index: c * 2 + r (c in 0..3, r in 0..1) -> W[k*n+i] with k=col, i=row
    
    def get_u_global(local_r, local_c, a_col_offset):
        # local_r in 0..1, local_c in 0..1
        # a_col_offset: 0 for A_L, 2 for A_R
        g_r = local_r
        g_c = a_col_offset + local_c
        return g_r * 4 + g_c

    def get_v_global(local_r, local_c, b_row_offset, b_col_offset):
        # local_r, local_c in 0..1
        # b_row_offset: 0 for Top, 2 for Bottom
        # b_col_offset: 0 for Left, 2 for Right
        g_r = b_row_offset + local_r
        g_c = b_col_offset + local_c
        return g_r * 4 + g_c

    def get_w_global(local_r, local_c, c_col_offset):
        # local_r, local_c in 0..1
        # c_col_offset: 0 for C_L, 2 for C_R
        g_r = local_r
        g_c = c_col_offset + local_c
        return g_c * 2 + g_r

    # List of block configurations: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    configs = [
        (0, 0, 0, 0), # A_L, B_TL -> C_L
        (2, 2, 0, 0), # A_R, B_BL -> C_L
        (0, 0, 2, 2), # A_L, B_TR -> C_R
        (2, 2, 2, 2), # A_R, B_BR -> C_R
    ]
    
    current_col = 0
    for cfg in configs:
        a_off, b_row_off, b_col_off, c_col_off = cfg
        
        for s_idx in range(7):
            col_idx = current_col + s_idx
            u_ind, u_coeff, v_ind, v_coeff, w_ind, w_coeff = strassen_defs[s_idx]
            
            # Add to U
            for i in range(len(u_ind)):
                lr = u_ind[i] // 2
                lc = u_ind[i] % 2
                g_idx = get_u_global(lr, lc, a_off)
                U[g_idx, col_idx] += u_coeff[i]
                
            # Add to V
            for i in range(len(v_ind)):
                lr = v_ind[i] // 2
                lc = v_ind[i] % 2
                g_idx = get_v_global(lr, lc, b_row_off, b_col_off)
                V[g_idx, col_idx] += v_coeff[i]
                
            # Add to W
            for i in range(len(w_ind)):
                lr = w_ind[i] // 2
                lc = w_ind[i] % 2
                g_idx = get_w_global(lr, lc, c_col_off)
                W[g_idx, col_idx] += w_coeff[i]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = 2, 4, 4
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    Cf = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            acc = 0.0
            for r in range(U.shape[1]):
                for j in range(m):
                    acc += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            Cf[i, k] = acc
            
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
