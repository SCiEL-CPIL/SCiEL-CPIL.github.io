import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 terms: (u_idx, u_coeff), (v_idx, v_coeff), (w_idx, w_coeff)
    # Local indices 0..3 for 2x2 matrices (row-major)
    # M1 = (X00+X11)(Y00+Y11) -> P00 += 1, P11 += 1
    # M2 = (X10+X11)Y00       -> P10 += 1, P11 -= 1
    # M3 = X00(Y01-Y11)       -> P01 += 1, P11 += 1
    # M4 = X11(Y10-Y00)       -> P00 += 1, P10 += 1
    # M5 = (X00+X01)Y11       -> P00 -= 1, P01 += 1
    # M6 = (X10-X00)(Y00+Y01) -> P11 += 1
    # M7 = (X01-X11)(Y10+Y11) -> P00 += 1
    
    strassen_terms = [
        # M1
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        # M3
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # M4
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (0, 1), (2, 1) ]),
        # M5
        ([ (0, 1), (1, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        # M6
        ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ], [ (3, 1) ]),
        # M7
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (0, 1) ])
    ]
    
    # Blocks:
    # A is 2x4. Split into A_left (cols 0-1) and A_right (cols 2-3).
    # B is 4x4. Split into B_top (rows 0-1) and B_bottom (rows 2-3).
    # C is 2x4. Split into C_left (cols 0-1) and C_right (cols 2-3).
    #
    # C_left = A_left * B_top_left + A_right * B_bottom_left
    # C_right = A_left * B_top_right + A_right * B_bottom_right
    #
    # This is 4 independent 2x2 multiplications in terms of variable support?
    # No, A_left is shared between C_left and C_right.
    # However, we can treat this as 4 separate 2x2x2 problems if we duplicate A?
    # No, that increases rank.
    #
    # The standard decomposition for 2x4 by 4x4 is to use 4 Strassen decompositions.
    # 1. A_left * B_top_left -> C_left_part1
    # 2. A_right * B_bottom_left -> C_left_part2
    # 3. A_left * B_top_right -> C_right_part1
    # 4. A_right * B_bottom_right -> C_right_part2
    #
    # Wait, if we just sum them, we need to add the results.
    # The tensor decomposition is linear.
    # C = C_part1 + C_part2 + C_part3 + C_part4.
    # Each C_part is a 2x2 matrix product.
    # The rank of a sum of tensors is at most the sum of their ranks.
    # Rank(M_2x2x2) = 7.
    # So Rank <= 4 * 7 = 28.
    #
    # We construct the 28 terms by iterating through the 4 blocks and 7 Strassen terms.
    
    # Block definitions:
    # (a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
    # A is 2x4. Rows 0-1. Cols 0-3.
    # B is 4x4. Rows 0-3. Cols 0-3.
    # C is 2x4. Rows 0-1. Cols 0-3.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    blocks = [
        # a_r, a_c, b_r, b_c, c_r, c_c
        (0, 0, 0, 0, 0, 0), # 1
        (0, 2, 2, 0, 0, 0), # 2
        (0, 0, 0, 2, 0, 2), # 3
        (0, 2, 2, 2, 0, 2)  # 4
    ]
    
    col_idx = 0
    for a_r, a_c, b_r, b_c, c_r, c_c in blocks:
        for u_list, v_list, w_list in strassen_terms:
            # Update U
            for local_idx, coeff in u_list:
                lr = local_idx // 2
                lc = local_idx % 2
                global_row = a_r + lr
                global_col = a_c + lc
                u_idx = global_row * m + global_col
                U[u_idx, col_idx] += coeff
            
            # Update V
            for local_idx, coeff in v_list:
                lr = local_idx // 2
                lc = local_idx % 2
                global_row = b_r + lr
                global_col = b_c + lc
                v_idx = global_row * p + global_col
                V[v_idx, col_idx] += coeff
            
            # Update W
            for local_idx, coeff in w_list:
                lr = local_idx // 2
                lc = local_idx % 2
                global_row = c_r + lr
                global_col = c_c + lc
                # W[k*n + i] where k is col, i is row
                w_idx = global_col * n + global_row
                W[w_idx, col_idx] += coeff
            
            col_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
