"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Strassen-based decomposition: split 6x6 into 2x2 blocks of 3x3
    # 7 Strassen steps, each with 27 schoolbook terms for 3x3 multiplication
    # Total rank = 7 * 27 = 189 (better than schoolbook 216)
    
    rank = 7 * 27
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Block indices:
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11
    
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C combination:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    steps = [
        # (A_coeffs: {block: sign}, B_coeffs: {block: sign})
        ({0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}),              # M2
        ({0: 1}, {1: 1, 3: -1}),             # M3
        ({3: 1}, {2: 1, 0: -1}),             # M4
        ({0: 1, 1: 1}, {3: 1}),              # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}),       # M7
    ]
    
    # For each step s, which C blocks does it contribute to and with what sign?
    c_contrib = [
        # step 0 (M1): C00(+1), C11(+1)
        {0: 1, 3: 1},
        # step 1 (M2): C10(+1), C11(-1)
        {2: 1, 3: -1},
        # step 2 (M3): C01(+1), C11(+1)
        {1: 1, 3: 1},
        # step 3 (M4): C00(-1), C10(+1)
        {0: -1, 2: 1},
        # step 4 (M5): C01(+1)
        {1: 1},
        # step 5 (M6): C00(+1)
        {0: 1},
        # step 6 (M7): C00(-1), C11(+1)
        {0: -1, 3: 1},
    ]
    
    # For each Strassen step s and each schoolbook term (i,j,k) in 3x3:
    # i = row in A block (0,1,2), j = col in A block / row in B block (0,1,2), k = col in B block (0,1,2)
    # U: for each A_block with sign a_sign, set U[row, col] += a_sign
    #   row = A_block_row * 3 + i, col_index = A_block_col * 3 + j
    # V: for each B_block with sign b_sign, set V[row, col] += b_sign
    #   row = B_block_row * 3 + j, col_index = B_block_col * 3 + k
    # W: for each C_block with sign c_sign, set W[row, col] += c_sign
    #   row = C_block_col * 3 + k, col_index = C_block_row * 3 + i
    #   (W uses trace convention: W[k*n+i, r])
    
    for s in range(7):
        a_coeffs = steps[s][0]
        b_coeffs = steps[s][1]
        c_coeffs = c_contrib[s]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = s * 27 + i * 9 + j * 3 + k
                    
                    # U: contributions from A blocks
                    for a_block, a_sign in a_coeffs.items():
                        a_row = (a_block // 2) * 3 + i
                        a_col = (a_block % 2) * 3 + j
                        U[a_row * m + a_col][col] += float(a_sign)
                    
                    # V: contributions from B blocks
                    for b_block, b_sign in b_coeffs.items():
                        b_row = (b_block // 2) * 3 + j
                        b_col = (b_block % 2) * 3 + k
                        V[b_row * p + b_col][col] += float(b_sign)
                    
                    # W: contributions to C blocks (trace convention: W[k*n+i])
                    for c_block, c_sign in c_coeffs.items():
                        c_row = (c_block // 2) * 3 + i
                        c_col = (c_block % 2) * 3 + k
                        W[c_col * n + c_row][col] += float(c_sign)
    
    # Check deadline
    if time.time() > DEADLINE:
        # Fallback to schoolbook if we're running out of time (shouldn't happen for this small case)
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
