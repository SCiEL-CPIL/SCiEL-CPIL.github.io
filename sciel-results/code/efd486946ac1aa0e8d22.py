import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    # We are targeting rank 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get index in U/V/W
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    # Define the 7 Strassen block products M_k
    # X_k and Y_k are 2x2 matrices of linear forms
    # Each linear form is a vector of length 16 (flattened 4x4 matrix)
    
    # We represent linear forms as lists of 16 coefficients
    # A is 4x4. Block A_rc is A[2r:2r+2, 2c:2c+2]
    # A_rc[i,j] is at index (2r+i)*4 + (2c+j)
    
    def get_block_coeff(r, c, i, j):
        # Returns a vector of length 16 with 1 at position (2r+i)*4 + (2c+j)
        vec = [0] * 16
        vec[(2*r + i) * 4 + (2*c + j)] = 1
        return vec
    
    # Define X_k and Y_k for k=0..6
    # X_k[i][j] and Y_k[i][j] are lists of linear forms (lists of 16 coeffs)
    
    X = [[None for _ in range(2)] for _ in range(2)]
    Y = [[None for _ in range(2)] for _ in range(2)]
    
    # M_1: (A_00 + A_11)(B_00 + B_11)
    # X_1 = A_00 + A_11
    # Y_1 = B_00 + B_11
    # M_2: (A_10 + A_11)B_00
    # M_3: A_00(B_01 - B_11)
    # M_4: A_11(B_10 - B_00)
    # M_5: (A_00 + A_01)B_11
    # M_6: (A_10 - A_00)(B_00 + B_01)
    # M_7: (A_01 - A_11)(B_10 + B_11)
    
    # Let's build the linear forms for each M_k
    # We will store them in a list of 7 items, each having X and Y 2x2 arrays of coeffs
    
    M_defs = []
    
    # k=0
    x = [[None, None], [None, None]]
    y = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            x[i][j] = [a + b for a, b in zip(get_block_coeff(0, 0, i, j), get_block_coeff(1, 1, i, j))]
            y[i][j] = [a + b for a, b in zip(get_block_coeff(0, 0, i, j), get_block_coeff(1, 1, i, j))]
    M_defs.append((x, y))
    
    # k=1
    x = [[None, None], [None, None]]
    y = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            x[i][j] = [a + b for a, b in zip(get_block_coeff(1, 0, i, j), get_block_coeff(1, 1, i, j))]
            y[i][j] = get_block_coeff(0, 0, i, j)
    M_defs.append((x, y))
    
    # k=2
    x = [[None, None], [None, None]]
    y = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            x[i][j] = get_block_coeff(0, 0, i, j)
            y[i][j] = [a - b for a, b in zip(get_block_coeff(0, 1, i, j), get_block_coeff(1, 1, i, j))]
    M_defs.append((x, y))
    
    # k=3
    x = [[None, None], [None, None]]
    y = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            x[i][j] = get_block_coeff(1, 1, i, j)
            y[i][j] = [a - b for a, b in zip(get_block_coeff(1, 0, i, j), get_block_coeff(0, 0, i, j))]
    M_defs.append((x, y))
    
    # k=4
    x = [[None, None], [None, None]]
    y = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            x[i][j] = [a + b for a, b in zip(get_block_coeff(0, 0, i, j), get_block_coeff(0, 1, i, j))]
            y[i][j] = get_block_coeff(1, 1, i, j)
    M_defs.append((x, y))
    
    # k=5
    x = [[None, None], [None, None]]
    y = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            x[i][j] = [a - b for a, b in zip(get_block_coeff(1, 0, i, j), get_block_coeff(0, 0, i, j))]
            y[i][j] = [a + b for a, b in zip(get_block_coeff(0, 0, i, j), get_block_coeff(0, 1, i, j))]
    M_defs.append((x, y))
    
    # k=6
    x = [[None, None], [None, None]]
    y = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            x[i][j] = [a - b for a, b in zip(get_block_coeff(0, 1, i, j), get_block_coeff(1, 1, i, j))]
            y[i][j] = [a + b for a, b in zip(get_block_coeff(1, 0, i, j), get_block_coeff(1, 1, i, j))]
    M_defs.append((x, y))
    
    # Coefficients of M_k in C blocks
    # C_00 = M_0 + M_3 - M_4 + M_6
    # C_01 = M_2 + M_4
    # C_10 = M_1 + M_3
    # C_11 = M_0 - M_1 + M_2 + M_5
    # Note: My previous mapping was:
    # M_1 (idx 0): C_00 +1, C_11 +1
    # M_2 (idx 1): C_10 +1, C_11 -1
    # M_3 (idx 2): C_01 +1, C_11 +1
    # M_4 (idx 3): C_00 +1, C_10 +1
    # M_5 (idx 4): C_00 -1, C_01 +1
    # M_6 (idx 5): C_11 +1
    # M_7 (idx 6): C_00 +1
    
    # Let's re-verify indices.
    # M_1 = M_defs[0]
    # M_2 = M_defs[1]
    # M_3 = M_defs[2]
    # M_4 = M_defs[3]
    # M_5 = M_defs[4]
    # M_6 = M_defs[5]
    # M_7 = M_defs[6]
    
    # C_00 = M_1 + M_4 - M_5 + M_7  -> 0, 3, 4, 6
    # C_01 = M_3 + M_5               -> 2, 4
    # C_10 = M_2 + M_4               -> 1, 3
    # C_11 = M_1 - M_2 + M_3 + M_6  -> 0, 1, 2, 5
    
    block_coeffs = {
        (0, 0): {0: 1, 3: 1, 4: -1, 6: 1},
        (0, 1): {2: 1, 4: 1},
        (1, 0): {1: 1, 3: 1},
        (1, 1): {0: 1, 1: -1, 2: 1, 5: 1}
    }
    
    # Coefficients of S_s in Z_k (2x2)
    # Z_00 = S_1 + S_4 - S_5 + S_7
    # Z_01 = S_3 + S_5
    # Z_10 = S_2 + S_4
    # Z_11 = S_1 - S_2 + S_3 + S_6
    # Indices 0-6 for S_1..S_7
    
    z_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # Z_00
        {2: 1, 4: 1},             # Z_01
        {1: 1, 3: 1},             # Z_10
        {0: 1, 1: -1, 2: 1, 5: 1} # Z_11
    ]
    
    # Strassen steps for 2x2 multiplication
    # S_1 = (x_00 + x_11)(y_00 + y_11)
    # S_2 = (x_10 + x_11)y_00
    # S_3 = x_00(y_01 - y_11)
    # S_4 = x_11(y_10 - y_00)
    # S_5 = (x_00 + x_01)y_11
    # S_6 = (x_10 - x_00)(y_00 + y_01)
    # S_7 = (x_01 - x_11)(y_10 + y_11)
    
    # We need to extract the linear forms for x and y parts for each S_s
    # For a given k, we have X_k and Y_k.
    
    col = 0
    for k in range(7):
        X_k, Y_k = M_defs[k]
        for s in range(7):
            # Determine linear forms Lx and Ly
            if s == 0:
                lx = [a + b for a, b in zip(X_k[0][0], X_k[1][1])]
                ly = [a + b for a, b in zip(Y_k[0][0], Y_k[1][1])]
            elif s == 1:
                lx = [a + b for a, b in zip(X_k[1][0], X_k[1][1])]
                ly = Y_k[0][0]
            elif s == 2:
                lx = X_k[0][0]
                ly = [a - b for a, b in zip(Y_k[0][1], Y_k[1][1])]
            elif s == 3:
                lx = X_k[1][1]
                ly = [a - b for a, b in zip(Y_k[1][0], Y_k[0][0])]
            elif s == 4:
                lx = [a + b for a, b in zip(X_k[0][0], X_k[0][1])]
                ly = Y_k[1][1]
            elif s == 5:
                lx = [a - b for a, b in zip(X_k[1][0], X_k[0][0])]
                ly = [a + b for a, b in zip(Y_k[0][0], Y_k[0][1])]
            elif s == 6:
                lx = [a - b for a, b in zip(X_k[0][1], X_k[1][1])]
                ly = [a + b for a, b in zip(Y_k[1][0], Y_k[1][1])]
            
            # Fill U and V
            for i in range(16):
                U[i][col] = lx[i]
                V[i][col] = ly[i]
            
            # Fill W
            # Iterate over blocks R, C
            for R in range(2):
                for C in range(2):
                    if k not in block_coeffs[(R, C)]:
                        continue
                    coeff_k = block_coeffs[(R, C)][k]
                    # Iterate over sub-block positions r, c
                    for r in range(2):
                        for c in range(2):
                            if s not in z_coeffs[r*2+c]:
                                continue
                            coeff_s = z_coeffs[r*2+c][s]
                            total_coeff = coeff_k * coeff_s
                            if total_coeff != 0:
                                # Global row i = 2R + r
                                # Global col j = 2C + c
                                i_global = 2*R + r
                                j_global = 2*C + c
                                idx = j_global * 4 + i_global
                                W[idx][col] += total_coeff
            
            col += 1
            
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    return {"U": U, "V": V, "W": W}
