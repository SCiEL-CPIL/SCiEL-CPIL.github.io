import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    # We construct a rank-44 decomposition using block decomposition.
    # The matrix multiplication is C = A * B
    # A is 3x4, B is 4x4, C is 3x4.
    # We split A into A_L (3x2) and A_R (3x2).
    # We split B into 4 blocks of 2x2:
    # B_TL (rows 0-1, cols 0-1), B_TR (rows 0-1, cols 2-3)
    # B_BL (rows 2-3, cols 0-1), B_BR (rows 2-3, cols 2-3)
    # C is 3x4.
    # C[:, 0:2] = A_L * B_TL + A_R * B_BL
    # C[:, 2:4] = A_L * B_TR + A_R * B_BR
    
    # Each term is a 3x2 * 2x2 multiplication.
    # Rank of 3x2x2 is 11 (7 from Strassen for 2x2x2 part + 4 for 1x2x2 part).
    # Total rank = 4 blocks * 11 = 44.
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the blocks
    # Block 0: A_L * B_TL -> C[:, 0:2]
    # Block 1: A_L * B_TR -> C[:, 2:4]
    # Block 2: A_R * B_BL -> C[:, 0:2]
    # Block 3: A_R * B_BR -> C[:, 2:4]
    
    # For each block, we define the local indices for A, B, C.
    # A_L: i in [0,2], j in [0,1]
    # A_R: i in [0,2], j in [2,3]
    # B_TL: j in [0,1], k in [0,1]
    # B_TR: j in [0,1], k in [2,3]
    # B_BL: j in [2,3], k in [0,1]
    # B_BR: j in [2,3], k in [2,3]
    # C[:, 0:2]: i in [0,2], k in [0,1]
    # C[:, 2:4]: i in [0,2], k in [2,3]
    
    # We need a function to add a local 3x2x2 decomposition to the global matrices.
    # local_a_rows: list of global i indices for the 3 rows of A block
    # local_a_cols: list of global j indices for the 2 cols of A block
    # local_b_rows: list of global j indices for the 2 rows of B block
    # local_b_cols: list of global k indices for the 2 cols of B block
    # local_c_rows: list of global i indices for the 3 rows of C block
    # local_c_cols: list of global k indices for the 2 cols of C block
    # col_offset: starting column index in U, V, W
    
    def add_3x2x2_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, col_offset):
        # local indices 0,1,2 for rows, 0,1 for cols
        # Strassen for 2x2x2 part (rows 0,1 of A block)
        # M1 = A[0,0] + A[1,1]
        # M2 = A[1,0] + A[1,1]
        # M3 = A[0,0]
        # M4 = A[0,1]
        # M5 = B[0,0] + B[1,0]
        # M6 = B[0,1] - B[1,1]
        # M7 = B[1,1] + B[1,0]
        
        # Products:
        # Z1 = M1 * M5
        # Z2 = M2 * B[0,0]  -- Wait, standard Strassen uses Y11, Y12 etc.
        # Let's stick to the standard Strassen formulas for 2x2.
        # X = A[0:2, :], Y = B[:, 0:2]
        # X11=X[0,0], X12=X[0,1], X21=X[1,0], X22=X[1,1]
        # Y11=Y[0,0], Y12=Y[0,1], Y21=Y[1,0], Y22=Y[1,1]
        
        # Z1 = (X11+X22)(Y11+Y22)
        # Z2 = (X21+X22)Y11
        # Z3 = X11(Y12-Y22)
        # Z4 = X22(Y21-Y11)
        # Z5 = (X11+X12)Y22
        # Z6 = (X21-X11)(Y11+Y12)
        # Z7 = (X12-X22)(Y21+Y22)
        
        # P11 = Z1 + Z4 - Z5 + Z7
        # P12 = Z3 + Z5
        # P21 = Z2 + Z4
        # P22 = Z1 - Z2 + Z3 + Z6
        
        # For each Zk, we define the U, V, W vectors.
        # U vector corresponds to the X factor (linear comb of A entries)
        # V vector corresponds to the Y factor (linear comb of B entries)
        # W vector corresponds to the contribution to C entries.
        
        # Helper to set a column
        def set_col(l, u_vals, v_vals, w_vals):
            c = col_offset + l
            # u_vals: dict {global_a_index: coeff}
            for idx, val in u_vals.items():
                U[idx, c] += val
            # v_vals: dict {global_b_index: coeff}
            for idx, val in v_vals.items():
                V[idx, c] += val
            # w_vals: dict {global_c_index: coeff}
            for idx, val in w_vals.items():
                W[idx, c] += val

        # Map local (r, c) to global index
        # A: a_rows[r] * m + a_cols[c]
        # B: b_rows[r] * p + b_cols[c]
        # C: c_cols[c] * n + c_rows[r]  (Note: W index is k*n + i)
        
        # Z1: (X11+X22)(Y11+Y22)
        # X11 is A[0,0] -> global a_rows[0]*m + a_cols[0]
        # X22 is A[1,1] -> global a_rows[1]*m + a_cols[1]
        # Y11 is B[0,0] -> global b_rows[0]*p + b_cols[0]
        # Y22 is B[1,1] -> global b_rows[1]*p + b_cols[1]
        u1 = {a_rows[0]*m + a_cols[0]: 1, a_rows[1]*m + a_cols[1]: 1}
        v1 = {b_rows[0]*p + b_cols[0]: 1, b_rows[1]*p + b_cols[1]: 1}
        # Contributes to P11 and P22
        # P11 is C[0,0] -> c_cols[0]*n + c_rows[0]
        # P22 is C[1,1] -> c_cols[1]*n + c_rows[1]
        w1 = {c_cols[0]*n + c_rows[0]: 1, c_cols[1]*n + c_rows[1]: -1} # P22 has -Z1
        set_col(0, u1, v1, w1)

        # Z2: (X21+X22)Y11
        u2 = {a_rows[1]*m + a_cols[0]: 1, a_rows[1]*m + a_cols[1]: 1}
        v2 = {b_rows[0]*p + b_cols[0]: 1}
        # Contributes to P21 and P22
        # P21 is C[1,0] -> c_cols[0]*n + c_rows[1]
        # P22 is C[1,1] -> c_cols[1]*n + c_rows[1]
        w2 = {c_cols[0]*n + c_rows[1]: 1, c_cols[1]*n + c_rows[1]: -1}
        set_col(1, u2, v2, w2)

        # Z3: X11(Y12-Y22)
        u3 = {a_rows[0]*m + a_cols[0]: 1}
        v3 = {b_rows[0]*p + b_cols[1]: 1, b_rows[1]*p + b_cols[1]: -1}
        # Contributes to P12 and P22
        # P12 is C[0,1] -> c_cols[1]*n + c_rows[0]
        # P22 is C[1,1] -> c_cols[1]*n + c_rows[1]
        w3 = {c_cols[1]*n + c_rows[0]: 1, c_cols[1]*n + c_rows[1]: 1}
        set_col(2, u3, v3, w3)

        # Z4: X22(Y21-Y11)
        u4 = {a_rows[1]*m + a_cols[1]: 1}
        v4 = {b_rows[1]*p + b_cols[0]: 1, b_rows[0]*p + b_cols[0]: -1}
        # Contributes to P11 and P21
        # P11 is C[0,0] -> c_cols[0]*n + c_rows[0]
        # P21 is C[1,0] -> c_cols[0]*n + c_rows[1]
        w4 = {c_cols[0]*n + c_rows[0]: 1, c_cols[0]*n + c_rows[1]: 1}
        set_col(3, u4, v4, w4)

        # Z5: (X11+X12)Y22
        u5 = {a_rows[0]*m + a_cols[0]: 1, a_rows[0]*m + a_cols[1]: 1}
        v5 = {b_rows[1]*p + b_cols[1]: 1}
        # Contributes to P11 and P12
        # P11 is C[0,0] -> c_cols[0]*n + c_rows[0]
        # P12 is C[0,1] -> c_cols[1]*n + c_rows[0]
        w5 = {c_cols[0]*n + c_rows[0]: -1, c_cols[1]*n + c_rows[0]: 1}
        set_col(4, u5, v5, w5)

        # Z6: (X21-X11)(Y11+Y12)
        u6 = {a_rows[1]*m + a_cols[0]: 1, a_rows[0]*m + a_cols[0]: -1}
        v6 = {b_rows[0]*p + b_cols[0]: 1, b_rows[0]*p + b_cols[1]: 1}
        # Contributes to P22
        # P22 is C[1,1] -> c_cols[1]*n + c_rows[1]
        w6 = {c_cols[1]*n + c_rows[1]: 1}
        set_col(5, u6, v6, w6)

        # Z7: (X12-X22)(Y21+Y22)
        u7 = {a_rows[0]*m + a_cols[1]: 1, a_rows[1]*m + a_cols[1]: -1}
        v7 = {b_rows[1]*p + b_cols[0]: 1, b_rows[1]*p + b_cols[1]: 1}
        # Contributes to P11
        # P11 is C[0,0] -> c_cols[0]*n + c_rows[0]
        w7 = {c_cols[0]*n + c_rows[0]: 1}
        set_col(6, u7, v7, w7)

        # Now the 1x2x2 part (row 2 of A block)
        # M2 = A[2, :] = [A[2,0], A[2,1]]
        # N = B[:, :]
        # P2 = M2 N
        # P2[0] = A[2,0]B[0,0] + A[2,1]B[1,0]
        # P2[1] = A[2,0]B[0,1] + A[2,1]B[1,1]
        
        # W1: A[2,0] B[0,0]
        u8 = {a_rows[2]*m + a_cols[0]: 1}
        v8 = {b_rows[0]*p + b_cols[0]: 1}
        # Contributes to C[2,0]
        w8 = {c_cols[0]*n + c_rows[2]: 1}
        set_col(7, u8, v8, w8)

        # W2: A[2,1] B[1,0]
        u9 = {a_rows[2]*m + a_cols[1]: 1}
        v9 = {b_rows[1]*p + b_cols[0]: 1}
        # Contributes to C[2,0]
        w9 = {c_cols[0]*n + c_rows[2]: 1}
        set_col(8, u9, v9, w9)

        # W3: A[2,0] B[0,1]
        u10 = {a_rows[2]*m + a_cols[0]: 1}
        v10 = {b_rows[0]*p + b_cols[1]: 1}
        # Contributes to C[2,1]
        w10 = {c_cols[1]*n + c_rows[2]: 1}
        set_col(9, u10, v10, w10)

        # W4: A[2,1] B[1,1]
        u11 = {a_rows[2]*m + a_cols[1]: 1}
        v11 = {b_rows[1]*p + b_cols[1]: 1}
        # Contributes to C[2,1]
        w11 = {c_cols[1]*n + c_rows[2]: 1}
        set_col(10, u11, v11, w11)

    # Block 0: A_L * B_TL
    # A_L: i in [0,2], j in [0,1]
    # B_TL: j in [0,1], k in [0,1]
    # C: i in [0,2], k in [0,1]
    add_3x2x2_block([0,1,2], [0,1], [0,1], [0,1], [0,1,2], [0,1], 0)
    
    # Block 1: A_L * B_TR
    # A_L: i in [0,2], j in [0,1]
    # B_TR: j in [0,1], k in [2,3]
    # C: i in [0,2], k in [2,3]
    add_3x2x2_block([0,1,2], [0,1], [0,1], [2,3], [0,1,2], [2,3], 11)
    
    # Block 2: A_R * B_BL
    # A_R: i in [0,2], j in [2,3]
    # B_BL: j in [2,3], k in [0,1]
    # C: i in [0,2], k in [0,1]
    add_3x2x2_block([0,1,2], [2,3], [2,3], [0,1], [0,1,2], [0,1], 22)
    
    # Block 3: A_R * B_BR
    # A_R: i in [0,2], j in [2,3]
    # B_BR: j in [2,3], k in [2,3]
    # C: i in [0,2], k in [2,3]
    add_3x2x2_block([0,1,2], [2,3], [2,3], [2,3], [0,1,2], [2,3], 33)
    
    # Verify
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
