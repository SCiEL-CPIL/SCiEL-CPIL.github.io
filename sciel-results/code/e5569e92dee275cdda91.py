"""Program for matrix multiplication tensor decomposition 2x2x13."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    """
    Returns factor matrices U, V, W for the 2x2x13 matrix multiplication tensor.
    Uses Strassen's algorithm (rank 7) for each 2x2x2 block and schoolbook for the last column.
    Total rank: 6 * 7 + 4 = 46.
    """
    n, m, p = 2, 2, 13
    
    # We will build rank-46 matrices
    total_rank = 6 * 7 + 4  # 42 + 4 = 46
    
    U = np.zeros((n * m, total_rank))  # shape (4, 46)
    V = np.zeros((m * p, total_rank))  # shape (26, 46)
    W = np.zeros((p * n, total_rank))  # shape (26, 46)
    
    # Strassen coefficients for 2x2x2
    # The 7 multiplications:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A01 - A00) * (B00 + B10)
    # M7 = (A10 - A11) * (B01 + B11)
    
    # U coefficients (rows are A00, A01, A10, A11 in order 0,1,2,3)
    U_coeff = np.array([
        [1, 0, 1, 1, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 1, 0],    # A01
        [0, 1, 0, 0, 0, 0, 1],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ], dtype=float)
    # U_coeff[i, j] is coefficient of A_i in M_j
    
    # V coefficients (for a single 2x2 block, rows are B00, B01, B10, B11 in order 0,1,2,3)
    V_coeff = np.array([
        [1, 1, 0, 0],    # M1: B00 + B11 -> row0=B00, row1=B01, row2=B10, row3=B11
        [1, 0, 0, 0],    # M2: B00
        [0, -1, 0, 1],   # M3: B01 - B11
        [0, 0, -1, 1],   # M4: B10 - B00
        [0, 0, 0, 1],    # M5: B11
        [1, 0, 1, 0],    # M6: B00 + B10
        [0, 1, 0, 1]     # M7: B01 + B11
    ], dtype=float).T  # Transpose so V_coeff[j, row] gives coefficient of B_row in M_j
    # Actually let's just define it directly: V_coeff[m, b] = coefficient of B_b in M_m
    # M1: B00+ B11 -> [1, 0, 0, 1]
    # M2: B00       -> [1, 0, 0, 0]
    # M3: B01- B11  -> [0, 1, 0, -1]
    # M4: B10- B00  -> [0, 0, 1, -1]
    # M5: B11       -> [0, 0, 0, 1]
    # M6: B00+ B10  -> [1, 0, 1, 0]
    # M7: B01+ B11  -> [0, 1, 0, 1]
    V_coeff = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=float)
    # V_coeff[m, b] = coeff of B_b in M_m
    
    # W coefficients (rows are C00, C01, C10, C11 in order 0,1,2,3)
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    W_coeff = np.array([
        [1, 0, 0, -1, 0, 1, 0],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, 0, -1, 0, 0, 0, 1]    # C11
    ], dtype=float)
    # W_coeff[c, m] = coeff of M_m in C_c
    
    # Process 6 blocks of 2 columns each
    for block in range(6):
        col_start = block * 2
        rank_offset = block * 7
        
        # U part: same for all blocks
        for r in range(7):
            for i in range(4):
                U[i, rank_offset + r] = U_coeff[i, r]
        
        # V part: rows are B_jk, where j in {0,1}, k in col_start, col_start+1
        # V row index = j * p + k = j * 13 + k
        # B00 -> j=0, k=col_start -> row 0*13 + col_start = col_start
        # B01 -> j=0, k=col_start+1 -> row 0*13 + col_start+1 = col_start+1
        # B10 -> j=1, k=col_start -> row 1*13 + col_start = 13 + col_start
        # B11 -> j=1, k=col_start+1 -> row 1*13 + col_start+1 = 13 + col_start + 1
        
        # For each multiplication r (0..6) and each B row b (0..3):
        for r in range(7):
            for b in range(4):
                coeff = V_coeff[r, b]
                if coeff != 0:
                    if b == 0:  # B00
                        v_row = col_start
                    elif b == 1:  # B01
                        v_row = col_start + 1
                    elif b == 2:  # B10
                        v_row = 13 + col_start
                    elif b == 3:  # B11
                        v_row = 13 + col_start + 1
                    V[v_row, rank_offset + r] = coeff
        
        # W part: rows are C_ik, where i in {0,1}, k in col_start, col_start+1
        # W row index = k * n + i = k * 2 + i
        # C00 -> i=0, k=col_start -> row col_start*2 + 0
        # C01 -> i=0, k=col_start+1 -> row (col_start+1)*2 + 0
        # C10 -> i=1, k=col_start -> row col_start*2 + 1
        # C11 -> i=1, k=col_start+1 -> row (col_start+1)*2 + 1
        
        for r in range(7):
            for c in range(4):
                coeff = W_coeff[c, r]
                if coeff != 0:
                    if c == 0:  # C00
                        w_row = col_start * 2
                    elif c == 1:  # C01
                        w_row = (col_start + 1) * 2
                    elif c == 2:  # C10
                        w_row = col_start * 2 + 1
                    elif c == 3:  # C11
                        w_row = (col_start + 1) * 2 + 1
                    W[w_row, rank_offset + r] = coeff
    
    # Last column (k=12), schoolbook 2x2x1, rank 4
    rank_offset = 42
    # M1 = A00 * B0,12
    # M2 = A01 * B1,12
    # M3 = A10 * B0,12
    # M4 = A11 * B1,12
    # C0,12 = M1 + M2
    # C1,12 = M3 + M4
    
    # U
    U[0, rank_offset + 0] = 1  # A00
    U[1, rank_offset + 1] = 1  # A01
    U[2, rank_offset + 2] = 1  # A10
    U[3, rank_offset + 3] = 1  # A11
    
    # V: B0,12 -> row 0*13+12 = 12, B1,12 -> row 1*13+12 = 25
    V[12, rank_offset + 0] = 1  # B0,12
    V[25, rank_offset + 1] = 1  # B1,12
    V[12, rank_offset + 2] = 1  # B0,12
    V[25, rank_offset + 3] = 1  # B1,12
    
    # W: C0,12 -> row 12*2+0 = 24, C1,12 -> row 12*2+1 = 25
    W[24, rank_offset + 0] = 1  # M1
    W[24, rank_offset + 1] = 1  # M2
    W[25, rank_offset + 2] = 1  # M3
    W[25, rank_offset + 3] = 1  # M4
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
