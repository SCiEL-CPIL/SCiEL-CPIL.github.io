import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # A (2x3) split into A_L (2x1) and A_R (2x2)
    # B (3x6) split into B_T (1x6) and B_B (2x6)
    # C = A_L B_T + A_R B_B
    
    # Part 1: A_L B_T (2x1 by 1x6)
    # Rank 12
    U1 = []
    V1 = []
    W1 = []
    for i in range(2):
        for k in range(6):
            u = [[0]*3 for _ in range(2)]
            u[i][0] = 1
            v = [[0]*6 for _ in range(3)]
            v[0][k] = 1
            w = [[0]*6 for _ in range(2)]
            w[i][k] = 1
            U1.append(u)
            V1.append(v)
            W1.append(w)
            
    # Part 2: A_R B_B (2x2 by 2x6)
    # Split B_B into 3 blocks of 2x2
    # Rank 3 * 7 = 21
    U2 = []
    V2 = []
    W2 = []
    
    # Strassen coefficients for 2x2 by 2x2
    # M1 = (a+d)(e+g)
    # M2 = (c+d)g
    # M3 = a(f-g)
    # M4 = d(h-f)
    # M5 = (a+b)e
    # M6 = (c+d)(h-f)
    # M7 = (a+b)(h-g)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 - M3
    
    # We need to map these to the global indices.
    # A_R corresponds to A[:, 1:3] (cols 1, 2 in 0-indexed)
    # B_B corresponds to B[1:3, :] (rows 1, 2 in 0-indexed)
    
    # For each of the 3 blocks of B_B (cols 0-1, 2-3, 4-5)
    for block in range(3):
        k_start = block * 2
        k_end = k_start + 2
        
        # M1
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start] = 1; v[1][k_start+1] = 1
        v[2][k_start] = 1; v[2][k_start+1] = 1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start] = 1; w[1][k_start] = 1
        U2.append(u); V2.append(v); W2.append(w)
        
        # M2
        u = [[0]*3 for _ in range(2)]
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[2][k_start+1] = 1
        w = [[0]*6 for _ in range(2)]
        w[1][k_start] = 1; w[1][k_start+1] = 1
        U2.append(u); V2.append(v); W2.append(w)
        
        # M3
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start+1] = 1; v[2][k_start+1] = -1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start+1] = 1
        U2.append(u); V2.append(v); W2.append(w)
        
        # M4
        u = [[0]*3 for _ in range(2)]
        u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[1][k_start] = -1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start] = 1; w[1][k_start] = 1
        U2.append(u); V2.append(v); W2.append(w)
        
        # M5
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start] = 1; v[1][k_start+1] = 1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start] = 1; w[0][k_start+1] = 1
        U2.append(u); V2.append(v); W2.append(w)
        
        # M6
        u = [[0]*3 for _ in range(2)]
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[2][k_start+1] = 1
        v[1][k_start] = -1; v[1][k_start+1] = -1
        w = [[0]*6 for _ in range(2)]
        w[1][k_start] = 1; w[1][k_start+1] = 1
        U2.append(u); V2.append(v); W2.append(w)
        
        # M7
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[2][k_start+1] = 1
        v[1][k_start] = -1; v[1][k_start+1] = -1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start] = 1; w[0][k_start+1] = 1
        U2.append(u); V2.append(v); W2.append(w)
        
        # Now we need to combine the W matrices for C11, C12, C21, C22
        # My U2, V2, W2 lists currently store the intermediate M terms.
        # But the output format requires U, V, W such that C = sum W_r * (U_r . A) * (V_r . B)
        # The Strassen decomposition provides 7 terms.
        # I added 7 terms to U2, V2, W2.
        # But I need to map them to C.
        # The W matrices I defined above are for the output C.
        # Wait, the W matrix in the decomposition is the coefficient for the term M_r.
        # So for M1, W should be such that it contributes to C11 and C22.
        
        # Let's restart the W construction for the Strassen part.
        # I will clear U2, V2, W2 and redo it properly.
        U2 = []
        V2 = []
        W2 = []
        
        # M1
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start] = 1; v[1][k_start+1] = 1
        v[2][k_start] = 1; v[2][k_start+1] = 1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start] = 1; w[1][k_start] = 1 # C11, C22
        # Wait, C11 gets M1, C22 gets M1.
        # So w[0][k_start] should be 1 for C11?
        # C is 2x6.
        # C[0, k_start] is C11 of the block.
        # C[1, k_start] is C21 of the block.
        # C[0, k_start+1] is C12 of the block.
        # C[1, k_start+1] is C22 of the block.
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M6 - M3
        
        # So for M1:
        # w[0][k_start] += 1
        # w[1][k_start+1] += 1
        
        # I will build the W matrices by summing contributions.
        # Initialize W for this block as 2x6 zeros.
        w_block = [[0]*6 for _ in range(2)]
        
        # M1
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start] = 1; v[1][k_start+1] = 1
        v[2][k_start] = 1; v[2][k_start+1] = 1
        U2.append(u); V2.append(v)
        w_block[0][k_start] += 1
        w_block[1][k_start+1] += 1
        
        # M2
        u = [[0]*3 for _ in range(2)]
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[2][k_start+1] = 1
        U2.append(u); V2.append(v)
        w_block[1][k_start] += 1
        w_block[1][k_start+1] -= 1
        
        # M3
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start+1] = 1; v[2][k_start+1] = -1
        U2.append(u); V2.append(v)
        w_block[0][k_start+1] += 1
        w_block[1][k_start+1] -= 1
        
        # M4
        u = [[0]*3 for _ in range(2)]
        u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[1][k_start] = -1
        U2.append(u); V2.append(v)
        w_block[0][k_start] += 1
        w_block[1][k_start] += 1
        
        # M5
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start] = 1; v[1][k_start+1] = 1
        U2.append(u); V2.append(v)
        w_block[0][k_start] -= 1
        w_block[0][k_start+1] += 1
        
        # M6
        u = [[0]*3 for _ in range(2)]
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[2][k_start+1] = 1
        v[1][k_start] = -1; v[1][k_start+1] = -1
        U2.append(u); V2.append(v)
        w_block[1][k_start+1] += 1
        
        # M7
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[2][k_start+1] = 1
        v[1][k_start] = -1; v[1][k_start+1] = -1
        U2.append(u); V2.append(v)
        w_block[0][k_start] += 1
        w_block[0][k_start+1] += 1
        
        # Add w_block to W2?
        # No, W2 is a list of W matrices.
        # Each term has its own W.
        # I accumulated w_block for all 7 terms.
        # But each term is a separate rank-1 tensor.
        # So I need 7 separate W matrices.
        
        # I will discard the w_block approach and create 7 W matrices.
        W2.append([[0]*6 for _ in range(2)]) # placeholder, will fill
        
        # Actually, I can just append the w matrices as I go.
        # Let's refactor.
        
    # This is getting messy. I will write a clean version.
    
    U = []
    V = []
    W = []
    
    # Part 1: A_L B_T
    for i in range(2):
        for k in range(6):
            u = [[0]*3 for _ in range(2)]
            u[i][0] = 1
            v = [[0]*6 for _ in range(3)]
            v[0][k] = 1
            w = [[0]*6 for _ in range(2)]
            w[i][k] = 1
            U.append(u)
            V.append(v)
            W.append(w)
            
    # Part 2: A_R B_B
    for block in range(3):
        k_start = block * 2
        k_end = k_start + 2
        
        # M1
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start] = 1; v[1][k_start+1] = 1
        v[2][k_start] = 1; v[2][k_start+1] = 1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start] = 1
        w[1][k_start+1] = 1
        U.append(u); V.append(v); W.append(w)
        
        # M2
        u = [[0]*3 for _ in range(2)]
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[2][k_start+1] = 1
        w = [[0]*6 for _ in range(2)]
        w[1][k_start] = 1
        w[1][k_start+1] = -1
        U.append(u); V.append(v); W.append(w)
        
        # M3
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start+1] = 1; v[2][k_start+1] = -1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start+1] = 1
        w[1][k_start+1] = -1
        U.append(u); V.append(v); W.append(w)
        
        # M4
        u = [[0]*3 for _ in range(2)]
        u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[1][k_start] = -1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start] = 1
        w[1][k_start] = 1
        U.append(u); V.append(v); W.append(w)
        
        # M5
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[1][k_start] = 1; v[1][k_start+1] = 1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start] = -1
        w[0][k_start+1] = 1
        U.append(u); V.append(v); W.append(w)
        
        # M6
        u = [[0]*3 for _ in range(2)]
        u[1][1] = 1; u[1][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[2][k_start+1] = 1
        v[1][k_start] = -1; v[1][k_start+1] = -1
        w = [[0]*6 for _ in range(2)]
        w[1][k_start+1] = 1
        U.append(u); V.append(v); W.append(w)
        
        # M7
        u = [[0]*3 for _ in range(2)]
        u[0][1] = 1; u[0][2] = 1
        v = [[0]*6 for _ in range(3)]
        v[2][k_start] = 1; v[2][k_start+1] = 1
        v[1][k_start] = -1; v[1][k_start+1] = -1
        w = [[0]*6 for _ in range(2)]
        w[0][k_start] = 1
        w[0][k_start+1] = 1
        U.append(u); V.append(v); W.append(w)
        
    return {"U": U, "V": V, "W": W}
