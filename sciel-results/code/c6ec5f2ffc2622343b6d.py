"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 108 via block decomposition
    # A (4x5) = A1 (4x2) + A2 (4x3)
    # B (5x6) = B1 (2x6) + B2 (3x6)
    # C = A1B1 + A2B2
    
    # Helper to get Strassen 2x2x2 factors
    def get_strassen_2x2x2():
        # U, V, W are lists of 7 vectors of length 4
        # Indices: 0=A00, 1=A01, 2=A10, 3=A11
        u = [
            [1, 0, 0, 1], # M1: A00 + A11
            [0, 0, 1, 1], # M2: A10 + A11
            [1, 0, 0, 0], # M3: A00
            [0, 0, 0, 1], # M4: A11
            [1, 1, 0, 0], # M5: A00 + A01
            [-1, 0, 1, 0],# M6: -A00 + A10
            [0, 1, 0, -1] # M7: A01 - A11
        ]
        v = [
            [1, 0, 0, 1], # M1: B00 + B11
            [1, 0, 0, 0], # M2: B00
            [0, 1, 0, -1],# M3: B01 - B11
            [-1, 0, 1, 0],# M4: -B00 + B10
            [0, 0, 0, 1], # M5: B11
            [1, 1, 0, 0], # M6: B00 + B01
            [0, 0, 1, 1]  # M7: B10 + B11
        ]
        w = [
            [1, 0, 0, 1], # C00 + M1, C11 + M1
            [0, 0, 1, 0], # C10 + M2
            [0, 1, 0, 1], # C01 + M3, C11 + M3
            [1, 0, 1, 0], # C00 + M4, C10 + M4
            [0, 1, 0, -1],# C01 + M5, C11 - M5
            [0, 0, 0, 1], # C11 + M6
            [1, 0, 0, 0]  # C00 + M7
        ]
        return u, v, w

    # Function to add a 2x2x2 block to global U, V, W
    # a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start
    # A is n x m, B is m x p, C is p x n
    # Local A is 2x2, B is 2x2, C is 2x2
    def add_strassen_block(U, V, W, a_r, a_c, b_r, b_c, c_r, c_c, n, m, p):
        u_local, v_local, w_local = get_strassen_2x2x2()
        for r in range(7):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            for li in range(2):
                for lj in range(2):
                    val = u_local[r][li * 2 + lj]
                    if val != 0:
                        u_vec[(a_r + li) * m + (a_c + lj)] = val
                        
            for li in range(2):
                for lj in range(2):
                    val = v_local[r][li * 2 + lj]
                    if val != 0:
                        v_vec[(b_r + li) * p + (b_c + lj)] = val
                        
            for li in range(2):
                for lj in range(2):
                    val = w_local[r][li * 2 + lj]
                    if val != 0:
                        w_vec[(c_c + lj) * n + (c_r + li)] = val # W[k*n+i] -> C[i,k] -> index k*n+i?
                        # Task says W[k*n+i, r]. C is p x n. i in 0..n-1, k in 0..p-1.
                        # C[i,k] corresponds to A row i, B col k.
                        # In w_local, li is row of C (0..1), lj is col of C (0..1).
                        # So global i = c_r + li, global k = c_c + lj.
                        # Index in W is k*n + i.
                        # My loop: w_vec[(c_c + lj) * n + (c_r + li)] is correct.
            
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)

    # Function to add scalar multiplication block (outer product)
    # A is n x 1, B is 1 x p
    def add_scalar_block(U, V, W, a_r, a_c, b_r, b_c, c_r, c_c, n, m, p):
        # A is 4x1, B is 1x6.
        # For each i in 0..3, k in 0..5:
        for i in range(4):
            for k in range(6):
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                # A element at (a_r + i, a_c)
                u_vec[(a_r + i) * m + a_c] = 1
                # B element at (b_r, b_c + k)
                v_vec[b_r * p + (b_c + k)] = 1
                # C element at (c_r + i, c_c + k) -> W index (c_c + k)*n + (c_r + i)
                w_vec[(c_c + k) * n + (c_r + i)] = 1
                
                U.append(u_vec)
                V.append(v_vec)
                W.append(w_vec)

    U = []
    V = []
    W = []

    # Part 1: A1 (4x2) * B1 (2x6)
    # Split A1 into two 2x2 blocks vertically: (0,0) and (2,0)
    # Split B1 into three 2x2 blocks horizontally: (0,0), (0,2), (0,4)
    # Block 1: A rows 0-1, cols 0-1. B rows 0-1, cols 0-1. C rows 0-1, cols 0-1.
    add_strassen_block(U, V, W, 0, 0, 0, 0, 0, 0, n, m, p)
    # Block 2: A rows 0-1, cols 0-1. B rows 0-1, cols 2-3. C rows 0-1, cols 2-3.
    add_strassen_block(U, V, W, 0, 0, 0, 2, 0, 2, n, m, p)
    # Block 3: A rows 0-1, cols 0-1. B rows 0-1, cols 4-5. C rows 0-1, cols 4-5.
    add_strassen_block(U, V, W, 0, 0, 0, 4, 0, 4, n, m, p)
    
    # Block 4: A rows 2-3, cols 0-1. B rows 0-1, cols 0-1. C rows 2-3, cols 0-1.
    add_strassen_block(U, V, W, 2, 0, 0, 0, 2, 0, n, m, p)
    # Block 5: A rows 2-3, cols 0-1. B rows 0-1, cols 2-3. C rows 2-3, cols 2-3.
    add_strassen_block(U, V, W, 2, 0, 0, 2, 2, 2, n, m, p)
    # Block 6: A rows 2-3, cols 0-1. B rows 0-1, cols 4-5. C rows 2-3, cols 4-5.
    add_strassen_block(U, V, W, 2, 0, 0, 4, 2, 4, n, m, p)
    
    # Part 2: A2 (4x3) * B2 (3x6)
    # Split A2 into A2a (4x2) and A2b (4x1)
    # Split B2 into B2a (2x6) and B2b (1x6)
    # A2a is cols 2-3 of A. B2a is rows 2-3 of B.
    # A2b is col 4 of A. B2b is row 4 of B.
    
    # Subpart 2a: A2a (4x2) * B2a (2x6)
    # Same structure as Part 1, but shifted.
    # A rows 0-1, cols 2-3. B rows 2-3, cols 0-1. C rows 0-1, cols 0-1.
    add_strassen_block(U, V, W, 0, 2, 2, 0, 0, 0, n, m, p)
    add_strassen_block(U, V, W, 0, 2, 2, 2, 0, 2, n, m, p)
    add_strassen_block(U, V, W, 0, 2, 2, 4, 0, 4, n, m, p)
    
    add_strassen_block(U, V, W, 2, 2, 2, 0, 2, 0, n, m, p)
    add_strassen_block(U, V, W, 2, 2, 2, 2, 2, 2, n, m, p)
    add_strassen_block(U, V, W, 2, 2, 2, 4, 2, 4, n, m, p)
    
    # Subpart 2b: A2b (4x1) * B2b (1x6)
    # A col 4. B row 4.
    add_scalar_block(U, V, W, 0, 4, 4, 0, 0, 0, n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
