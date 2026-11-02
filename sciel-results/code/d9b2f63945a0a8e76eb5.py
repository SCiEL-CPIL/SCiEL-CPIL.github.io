"""Matrix multiplication 6x6x6 - rank-minimal tensor decomposition via block Strassen."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: Partition each 6x6 matrix into a 3x3 grid of 2x2 blocks.
    For each result block C[i,k], compute sum_j A[i,j] * B[j,k] using
    Strassen's rank-7 algorithm for 2x2 matrix multiplication.
    
    Total rank = 3 * 3 * 3 * 7 = 189 (vs 216 for schoolbook).
    
    Tensor convention: C[k*n+i] = sum_r U[i*m+r'] * V[r'*p+k] ... 
    Actually per task: W[k*n+i, r] means output index is (col, row) of C.
    So W row index = c_col * n + c_row where c_col, c_row are C indices.
    U row index = a_row * m + a_col (A is n x m)
    V row index = b_row * p + b_col (B is m x p)
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    nb = 3  # number of blocks per side
    bs = 2  # block size
    
    rank = nb * nb * nb * 7  # 189
    
    # Initialize factor matrices as lists of lists (will convert to numpy at end if needed)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication:
    # Given X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M0 = (x00 + x11) * (y00 + y11)
    # M1 = (x10 + x11) * y00
    # M2 = x00 * (y01 - y11)
    # M3 = x11 * (y10 - y00)
    # M4 = (x00 + x01) * y11
    # M5 = (x10 - x00) * (y00 + y01)
    # M6 = (x01 - x11) * (y10 + y11)
    #
    # Result Z = X*Y:
    # Z00 = M0 + M3 - M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M3
    # Z11 = M0 - M1 + M2 + M5
    
    # For each Strassen term s, define:
    #   x_factors: dict from local (row, col) of X to coefficient
    #   y_factors: dict from local (row, col) of Y to coefficient
    #   z_factors: dict from local (row, col) of Z to coefficient (how M_s contributes to Z)
    
    strassen_terms = [
        # s=0: M0 = (x00+x11)*(y00+y11), contributes to Z00 (+1), Z11 (+1)
        ({(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1},
         {(0,0): 1, (1,1): 1}),
        # s=1: M1 = (x10+x11)*y00, contributes to Z10 (+1), Z11 (-1)
        ({(1,0): 1, (1,1): 1},
         {(0,0): 1},
         {(1,0): 1, (1,1): -1}),
        # s=2: M2 = x00*(y01-y11), contributes to Z01 (+1), Z11 (+1)
        ({(0,0): 1},
         {(0,1): 1, (1,1): -1},
         {(0,1): 1, (1,1): 1}),
        # s=3: M3 = x11*(y10-y00), contributes to Z00 (+1), Z10 (+1)
        ({(1,1): 1},
         {(1,0): 1, (0,0): -1},
         {(0,0): 1, (1,0): 1}),
        # s=4: M4 = (x00+x01)*y11, contributes to Z00 (-1), Z01 (+1)
        ({(0,0): 1, (0,1): 1},
         {(1,1): 1},
         {(0,0): -1, (0,1): 1}),
        # s=5: M5 = (x10-x00)*(y00+y01), contributes to Z11 (+1)
        ({(1,0): 1, (0,0): -1},
         {(0,0): 1, (0,1): 1},
         {(1,1): 1}),
        # s=6: M6 = (x01-x11)*(y10+y11), contributes to Z00 (+1)
        ({(0,1): 1, (1,1): -1},
         {(1,0): 1, (1,1): 1},
         {(0,0): 1}),
    ]
    
    col = 0
    for i_blk in range(nb):
        for k_blk in range(nb):
            for j_blk in range(nb):
                for s in range(7):
                    x_factors, y_factors, z_factors = strassen_terms[s]
                    
                    # Fill U: A block at (i_blk, j_blk), local element (lr_a, lc_a)
                    for (lr_a, lc_a), coeff in x_factors.items():
                        a_row = i_blk * bs + lr_a
                        a_col = j_blk * bs + lc_a
                        U[a_row * m + a_col][col] += coeff
                    
                    # Fill V: B block at (j_blk, k_blk), local element (lr_b, lc_b)
                    for (lr_b, lc_b), coeff in y_factors.items():
                        b_row = j_blk * bs + lr_b
                        b_col = k_blk * bs + lc_b
                        V[b_row * p + b_col][col] += coeff
                    
                    # Fill W: C block at (i_blk, k_blk), local element (lr_c, lc_c)
                    # W index convention: W[k*n+i, r] => row = c_col * n + c_row
                    for (lr_c, lc_c), coeff in z_factors.items():
                        c_row = i_blk * bs + lr_c
                        c_col = k_blk * bs + lc_c
                        W[c_col * n + c_row][col] += coeff
                    
                    col += 1
    
    # Convert to numpy arrays for efficiency
    import numpy as np
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    return {"U": U_arr, "V": V_arr, "W": W_arr}
# EVOLVE-BLOCK-END
