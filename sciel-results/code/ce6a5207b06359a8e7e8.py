import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    
    # Define Strassen terms for 2x2 matrix multiplication
    # A_coeffs[s]: list of (index_in_A, coeff)
    # B_coeffs[s]: list of (row_in_B, block_col_in_B, coeff)
    # W_coeffs[s]: tuple (coeff_C00, coeff_C01, coeff_C10, coeff_C11)
    
    A_coeffs = [
        [(0, 1), (3, 1)],    # M1: A00 + A11
        [(2, 1), (3, 1)],    # M2: A10 + A11
        [(0, 1)],             # M3: A00
        [(3, 1)],             # M4: A11
        [(0, 1), (1, 1)],    # M5: A00 + A01
        [(2, 1), (0, -1)],   # M6: A10 - A00
        [(1, 1), (3, -1)],   # M7: A01 - A11
    ]
    
    # B is split into two 2x5 blocks: B_L (cols 0-4) and B_R (cols 5-9)
    # block_col 0 -> cols 0-4, block_col 1 -> cols 5-9
    B_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1: B[0,0] + B[1,1]
        [(0, 0, 1)],               # M2: B[0,0]
        [(0, 1, 1), (1, 1, -1)],  # M3: B[0,1] - B[1,1]
        [(1, 0, 1), (0, 0, -1)],  # M4: B[1,0] - B[0,0]
        [(1, 1, 1)],               # M5: B[1,1]
        [(0, 0, 1), (0, 1, 1)],   # M6: B[0,0] + B[0,1]
        [(1, 0, 1), (1, 1, 1)],   # M7: B[1,0] + B[1,1]
    ]
    
    # Contributions to C blocks: C[0,0], C[0,1], C[1,0], C[1,1]
    # where C[i,j] is a 1x5 vector
    W_coeffs = [
        (1, 0, 0, 1),    # M1
        (0, 0, 1, -1),   # M2
        (0, 1, 0, 1),    # M3
        (1, 0, 1, 0),    # M4
        (-1, 1, 0, 0),   # M5
        (0, 0, 0, 1),    # M6
        (1, 0, 0, 0),    # M7
    ]
    
    rank = 35
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for s in range(7):
        a_list = A_coeffs[s]
        b_list = B_coeffs[s]
        w_c00, w_c01, w_c10, w_c11 = W_coeffs[s]
        
        for t in range(5):
            # Fill U
            for a_idx, a_c in a_list:
                U[a_idx, col] = a_c
            
            # Fill V
            for b_row, b_block, b_c in b_list:
                b_col = b_block * 5 + t
                V[b_row * p + b_col, col] = b_c
            
            # Fill W
            # C[0, t] is at index t*n + 0
            # C[0, 5+t] is at index (5+t)*n + 0
            # C[1, t] is at index t*n + 1
            # C[1, 5+t] is at index (5+t)*n + 1
            
            if w_c00 != 0:
                W[t * n + 0, col] = w_c00
            if w_c01 != 0:
                W[(5 + t) * n + 0, col] = w_c01
            if w_c10 != 0:
                W[t * n + 1, col] = w_c10
            if w_c11 != 0:
                W[(5 + t) * n + 1, col] = w_c11
            
            col += 1
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
