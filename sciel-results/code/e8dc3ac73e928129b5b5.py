"""Matrix multiplication 2,2,16 - rank-minimal exact tensor decomposition."""

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-56 exact tensor decomposition for M_{2,2,16}.
    
    Strategy: Group the 16 columns of B into 8 blocks of 2 columns each.
    For each block r (0..7), use a 7-product algorithm (based on Strassen-like
    additions/subtractions) to compute the 4 output entries C_{i, 2r} and C_{i, 2r+1}
    for i in {0, 1}.
    
    The 7 products per block are:
    M_1 = (A_11 + A_22) * (B_{0, 2r} + B_{1, 2r+1})
    M_2 = (A_21 + A_22) * B_{1, 2r+1}
    M_3 = A_11 * (B_{0, 2r+1} - B_{1, 2r})
    M_4 = A_22 * (B_{1, 2r} - B_{0, 2r})
    M_5 = (A_11 + A_12) * B_{0, 2r}
    M_6 = (A_21 + A_12) * (B_{1, 2r+1} - B_{0, 2r+1})
    M_7 = (A_11 - A_21) * (B_{0, 2r} + B_{1, 2r})
    
    Reconstruction:
    C_{0, 2r}   = M_1 + M_4 - M_5 + M_7
    C_{0, 2r+1} = M_3 + M_5
    C_{1, 2r}   = M_2 + M_4
    C_{1, 2r+1} = M_1 - M_2 + M_3 + M_6
    
    Total rank = 8 blocks * 7 products/block = 56.
    """
    import numpy as np

    n, m, p = 2, 2, 16
    rank = 56

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # U indices: row = i*m + j, where i is row of A (0,1), j is col of A (0,1)
    # So: U[0] = A_11, U[1] = A_12, U[2] = A_21, U[3] = A_22
    # V indices: row = j*p + k, where j is row of B (0,1), k is col of B (0..15)
    # W indices: row = k*n + i, where k is col of C (0..15), i is row of C (0,1)

    # For each block r (0..7) and product k (0..6):
    for r in range(8):
        for k in range(7):
            col = r * 7 + k
            
            # Determine coefficients for U (A factors)
            u_coeffs = [0, 0, 0, 0]  # A_11, A_12, A_21, A_22
            
            if k == 0:
                # M_1 = (A_11 + A_22) * ...
                u_coeffs[0] = 1  # A_11
                u_coeffs[3] = 1  # A_22
            elif k == 1:
                # M_2 = (A_21 + A_22) * ...
                u_coeffs[2] = 1  # A_21
                u_coeffs[3] = 1  # A_22
            elif k == 2:
                # M_3 = A_11 * ...
                u_coeffs[0] = 1  # A_11
            elif k == 3:
                # M_4 = A_22 * ...
                u_coeffs[3] = 1  # A_22
            elif k == 4:
                # M_5 = (A_11 + A_12) * ...
                u_coeffs[0] = 1  # A_11
                u_coeffs[1] = 1  # A_12
            elif k == 5:
                # M_6 = (A_21 + A_12) * ...
                u_coeffs[1] = 1  # A_12
                u_coeffs[2] = 1  # A_21
            elif k == 6:
                # M_7 = (A_11 - A_21) * ...
                u_coeffs[0] = 1  # A_11
                u_coeffs[2] = -1  # A_21
            
            for idx, val in enumerate(u_coeffs):
                U[idx, col] = val

            # Determine coefficients for V (B factors)
            # V index: j*p + k_col, where j is row of B (0,1), k_col is col of B
            v_coeffs = {}  # (row, col) -> coefficient
            
            if k == 0:
                # ... * (B_{0, 2r} + B_{1, 2r+1})
                v_coeffs[(0, 2*r)] = 1
                v_coeffs[(1, 2*r+1)] = 1
            elif k == 1:
                # ... * B_{1, 2r+1}
                v_coeffs[(1, 2*r+1)] = 1
            elif k == 2:
                # ... * (B_{0, 2r+1} - B_{1, 2r})
                v_coeffs[(0, 2*r+1)] = 1
                v_coeffs[(1, 2*r)] = -1
            elif k == 3:
                # ... * (B_{1, 2r} - B_{0, 2r})
                v_coeffs[(1, 2*r)] = 1
                v_coeffs[(0, 2*r)] = -1
            elif k == 4:
                # ... * B_{0, 2r}
                v_coeffs[(0, 2*r)] = 1
            elif k == 5:
                # ... * (B_{1, 2r+1} - B_{0, 2r+1})
                v_coeffs[(1, 2*r+1)] = 1
                v_coeffs[(0, 2*r+1)] = -1
            elif k == 6:
                # ... * (B_{0, 2r} + B_{1, 2r})
                v_coeffs[(0, 2*r)] = 1
                v_coeffs[(1, 2*r)] = 1
            
            for (row, col_b), val in v_coeffs.items():
                V[row * p + col_b, col] = val

            # Determine coefficients for W (C reconstruction)
            # W index: k_c * n + i, where k_c is col of C (0..15), i is row of C (0,1)
            w_coeffs = {}  # (col_c, row_c) -> coefficient
            
            if k == 0:
                # M_1 appears in: C_{0, 2r} (+1), C_{1, 2r+1} (+1)
                w_coeffs[(2*r, 0)] = 1
                w_coeffs[(2*r+1, 1)] = 1
            elif k == 1:
                # M_2 appears in: C_{1, 2r} (+1), C_{1, 2r+1} (-1)
                w_coeffs[(2*r, 1)] = 1
                w_coeffs[(2*r+1, 1)] = -1
            elif k == 2:
                # M_3 appears in: C_{0, 2r+1} (+1), C_{1, 2r+1} (+1)
                w_coeffs[(2*r+1, 0)] = 1
                w_coeffs[(2*r+1, 1)] = 1
            elif k == 3:
                # M_4 appears in: C_{0, 2r} (+1), C_{1, 2r} (+1)
                w_coeffs[(2*r, 0)] = 1
                w_coeffs[(2*r, 1)] = 1
            elif k == 4:
                # M_5 appears in: C_{0, 2r} (-1), C_{0, 2r+1} (+1)
                w_coeffs[(2*r, 0)] = -1
                w_coeffs[(2*r+1, 0)] = 1
            elif k == 5:
                # M_6 appears in: C_{1, 2r+1} (+1)
                w_coeffs[(2*r+1, 1)] = 1
            elif k == 6:
                # M_7 appears in: C_{0, 2r} (+1)
                w_coeffs[(2*r, 0)] = 1
            
            for (col_c, row_c), val in w_coeffs.items():
                W[col_c * n + row_c, col] = val

    # Convert to lists of lists as expected by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
