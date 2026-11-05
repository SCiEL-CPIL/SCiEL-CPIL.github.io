"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
        # Local indices 0,1 for 2x2 blocks
        # U coeffs for M1..M7
        u_coeffs = [
            [1, 0, 0, 1], # M1: A00 + A11
            [0, 0, 1, 1], # M2: A10 + A11
            [1, 0, 0, 0], # M3: A00
            [0, 0, 0, 1], # M4: A11
            [1, 1, 0, 0], # M5: A00 + A10
            [-1, 0, 1, 0],# M6: A10 - A00
            [0, 1, 0, -1] # M7: A01 - A11
        ]
        # V coeffs for M1..M7
        v_coeffs = [
            [1, 0, 0, 1], # M1: B00 + B11
            [1, 0, 0, 0], # M2: B00
            [0, 0, 1, -1],# M3: B10 - B11
            [0, 1, -1, 0],# M4: B01 - B00
            [0, 0, 0, 1], # M5: B11
            [1, 0, 1, 0], # M6: B00 + B10
            [0, 1, 0, 1]  # M7: B11 + B01
        ]
        # W coeffs for M1..M7
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        w_coeffs = [
            [1, 0, 0, 1],  # M1
            [-1, 0, 1, 0], # M2
            [0, 1, 0, 1],  # M3
            [1, 0, 1, 0],  # M4
            [-1, 1, 0, 0], # M5
            [0, 0, 0, 1],  # M6
            [1, 0, 0, 0]   # M7
        ]
        
        for t in range(7):
            col = offset + t
            # Fill U
            for lr in range(2):
                for lc in range(2):
                    val = u_coeffs[t][lr * 2 + lc]
                    if val != 0:
                        global_row = a_rows[lr] * m + a_cols[lc]
                        U[global_row][col] += val
                        
            # Fill V
            for lr in range(2):
                for lc in range(2):
                    val = v_coeffs[t][lr * 2 + lc]
                    if val != 0:
                        global_row = b_rows[lr] * p + b_cols[lc]
                        V[global_row][col] += val
                        
            # Fill W
            for lr in range(2):
                for lc in range(2):
                    val = w_coeffs[t][lr * 2 + lc]
                    if val != 0:
                        global_row = c_cols[lc] * n + c_rows[lr]
                        W[global_row][col] += val

    # Top 2 rows of A (rows 0, 1)
    # Block 1: A cols 0,1; B rows 0,1; C cols 0,1
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 0)
    # Block 2: A cols 2,3; B rows 2,3; C cols 0,1
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1], 7)
    # Block 3: A cols 0,1; B rows 0,1; C cols 2,3
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], 14)
    # Block 4: A cols 2,3; B rows 2,3; C cols 2,3
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3], 21)
    
    # Bottom row of A (row 2)
    # Schoolbook: 4x4 = 16 terms
    offset = 28
    for j in range(m):
        for k in range(p):
            col = offset + j * p + k
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
