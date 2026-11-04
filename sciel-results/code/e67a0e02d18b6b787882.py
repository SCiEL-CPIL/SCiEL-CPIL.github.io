import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U shape (n*m, rank) = (8, rank)
    # V shape (m*p, rank) = (16, rank)
    # W shape (p*n, rank) = (8, rank)
    
    # Strassen rank for 2x2x2 is 7
    # We have 4 such blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A indices in 2x2 block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices in 2x2 block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices in 2x2 block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Coefficients for U (A part) for each of the 7 multiplications
    # U_a[m] is coefficient of A_local[m] in M
    U_coeffs = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    # Coefficients for V (B part) for each of the 7 multiplications
    V_coeffs = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, 0, 1, -1],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    # Coefficients for W (C part) for each of the 7 multiplications
    # C_local[0] = C00, C_local[1] = C01, C_local[2] = C10, C_local[3] = C11
    W_coeffs = [
        [1, 0, 0, 0], # M1 -> C00
        [0, 0, 0, -1],# M2 -> C11 (-1)
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> C00(-1), C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # Define the 4 blocks
    # Block 1: A[:, 0:2], B[0:2, 0:2] -> C[:, 0:2]
    # Block 2: A[:, 2:4], B[2:4, 0:2] -> C[:, 0:2]
    # Block 3: A[:, 0:2], B[0:2, 2:4] -> C[:, 2:4]
    # Block 4: A[:, 2:4], B[2:4, 2:4] -> C[:, 2:4]
    
    # Mapping of local indices to global indices
    # A global index: i * m + j = i * 4 + j
    # B global index: j * p + k = j * 4 + k
    # C global index: k * n + i = k * 2 + i (where k is col of C, i is row of C)
    
    blocks = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset)
        # A is 2x4. Local A is 2x2.
        # a_row_offset is 0 for both.
        # a_col_offset is 0 or 2.
        # B is 4x4. Local B is 2x2.
        # b_row_offset is 0 or 2.
        # b_col_offset is 0 or 2.
        # C is 2x4. Local C is 2x2.
        # c_col_offset is 0 or 2.
        
        (0, 0, 0, 0, 0), # Block 1
        (0, 2, 2, 0, 0), # Block 2
        (0, 0, 0, 2, 2), # Block 3
        (0, 2, 2, 2, 2)  # Block 4
    ]
    
    col = 0
    for a_r_off, a_c_off, b_r_off, b_c_off, c_c_off in blocks:
        for s in range(7):
            # U coefficients
            for la in range(4):
                coeff = U_coeffs[s][la]
                if coeff != 0:
                    # Local A index la: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    li, lj = (la // 2, la % 2)
                    gi = a_r_off + li
                    gj = a_c_off + lj
                    g_idx = gi * m + gj
                    U[g_idx][col] += coeff
            
            # V coefficients
            for lb in range(4):
                coeff = V_coeffs[s][lb]
                if coeff != 0:
                    # Local B index lb: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    li, lj = (lb // 2, lb % 2)
                    gi = b_r_off + li
                    gj = b_c_off + lj
                    g_idx = gi * p + gj
                    V[g_idx][col] += coeff
            
            # W coefficients
            for lc in range(4):
                coeff = W_coeffs[s][lc]
                if coeff != 0:
                    # Local C index lc: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    li, lj = (lc // 2, lc % 2)
                    gi = li # row of C
                    gj = c_c_off + lj # col of C
                    # W index is gj * n + gi
                    g_idx = gj * n + gi
                    W[g_idx][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
