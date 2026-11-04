"""
Matrix multiplication 2x4 x 4x4 with rank-minimal decomposition.

Uses 4 Strassen 2x2x2 blocks, each of rank 7, for total rank 28.
Convention: W[k*n + i, r] where k is the column index of C, i is the row index.
"""
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Strassen 2x2x2 decomposition: 7 multiplications
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each 2x2 block, define the 7 Strassen factors as vectors of length 4
    # U vectors (for A factors), V vectors (for B factors), W vectors (for C outputs)
    
    # Strassen factors in order M1..M7
    # Each factor is a 4-tuple (A00, A01, A10, A11) for U,
    #                       (B00, B01, B10, B11) for V,
    #                       (C00, C01, C10, C11) for W
    
    # M1: (A00+A11)(B00+B11) -> C00
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 0]
    
    # M2: (A10+A11)B00 -> C10, C11
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1]
    
    # M3: A00(B01-B11) -> C01, C11
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]
    
    # M4: A11(B10-B00) -> C00, C10
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0]
    
    # M5: (A00+A01)B11 -> C00, C01
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    
    # M6: (A10-A00)(B00+B01) -> C11
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]
    
    # M7: (A01-A11)(B10+B11) -> C00
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]
    
    strassen_u = [u1, u2, u3, u4, u5, u6, u7]
    strassen_v = [v1, v2, v3, v4, v5, v6, v7]
    strassen_w = [w1, w2, w3, w4, w5, w6, w7]
    
    rank = 28  # 4 blocks * 7 factors
    
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Block structure:
    # A is 2x4, split into A_L = A[:, 0:2] and A_R = A[:, 2:4]
    # B is 4x4, split into B_L = B[0:2, :] and B_R = B[2:4, :]
    # C is 2x4, split into C_L = C[:, 0:2] and C_R = C[:, 2:4]
    #
    # Block 1: A_L * B_L -> C_L (rows 0-1 of A, cols 0-1 of B, rows 0-1 of C)
    #   A indices: (i, j) where i in {0,1}, j in {0,1} -> global index i*4 + j
    #   B indices: (k, l) where k in {0,1}, l in {0,1,2,3} -> global index k*4 + l
    #   C indices: (i, k) where i in {0,1}, k in {0,1} -> W index k*2 + i
    #
    # Block 2: A_L * B_R -> C_R (rows 0-1 of A, cols 2-3 of B, rows 0-1 of C)
    #   A indices: (i, j) where i in {0,1}, j in {0,1} -> global index i*4 + j
    #   B indices: (k, l) where k in {0,1}, l in {2,3} -> global index k*4 + l
    #   C indices: (i, k) where i in {0,1}, k in {0,1} -> W index (k+2)*2 + i
    #
    # Block 3: A_R * B_L -> C_L (rows 0-1 of A, cols 2-3 of A, rows 0-1 of B, rows 0-1 of C)
    #   A indices: (i, j) where i in {0,1}, j in {2,3} -> global index i*4 + j
    #   B indices: (k, l) where k in {0,1}, l in {0,1} -> global index k*4 + l
    #   C indices: (i, k) where i in {0,1}, k in {0,1} -> W index k*2 + i
    #
    # Block 4: A_R * B_R -> C_R (rows 0-1 of A, cols 2-3 of A, rows 2-3 of B, rows 0-1 of C)
    #   A indices: (i, j) where i in {0,1}, j in {2,3} -> global index i*4 + j
    #   B indices: (k, l) where k in {0,1}, l in {2,3} -> global index k*4 + l
    #   C indices: (i, k) where i in {0,1}, k in {0,1} -> W index (k+2)*2 + i
    
    # For each block, map local 2x2 indices to global indices
    # Block parameters: (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    blocks = [
        # (a_col_offset, b_col_offset, c_col_offset)
        (0, 0, 0),   # Block 1: A_L * B_L -> C_L
        (0, 2, 2),   # Block 2: A_L * B_R -> C_R
        (2, 0, 0),   # Block 3: A_R * B_L -> C_L
        (2, 2, 2),   # Block 4: A_R * B_R -> C_R
    ]
    
    for block_idx, (a_col_off, b_col_off, c_col_off) in enumerate(blocks):
        base_col = block_idx * 7
        for s in range(7):
            col = base_col + s
            u_vec = strassen_u[s]
            v_vec = strassen_v[s]
            w_vec = strassen_w[s]
            
            # Map local A indices (i, j) to global: i*4 + (j + a_col_off)
            for local_i in range(2):
                for local_j in range(2):
                    global_a_idx = local_i * 4 + (local_j + a_col_off)
                    U[global_a_idx][col] = u_vec[local_i * 2 + local_j]
            
            # Map local B indices (k, l) to global: k*4 + (l + b_col_off)
            for local_k in range(2):
                for local_l in range(2):
                    global_b_idx = local_k * 4 + (local_l + b_col_off)
                    V[global_b_idx][col] = v_vec[local_k * 2 + local_l]
            
            # Map local C indices (i, k) to W index: (k + c_col_off)*2 + i
            for local_i in range(2):
                for local_k in range(2):
                    w_idx = (local_k + c_col_off) * 2 + local_i
                    W[w_idx][col] = w_vec[local_i * 2 + local_k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
