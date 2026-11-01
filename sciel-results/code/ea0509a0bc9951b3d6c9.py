"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition.
Uses Strassen's algorithm on 7 independent 2x2 blocks to achieve rank 49.
"""

def construct():
    n, m, p = 2, 2, 14
    num_blocks = p // 2  # 7 blocks of 2x2
    rank = num_blocks * 7  # 49
    
    # Strassen's 7 linear forms for 2x2 matrix multiplication:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A coefficients: [a, b, c, d] for each Strassen term
    A_coeffs = [
        [1, 0, 0, 1],   # M1: (a+d)
        [0, 0, 1, 1],   # M2: (c+d)
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: (a+b)
        [-1, 0, 1, 0],  # M6: (c-a)
        [0, 1, 0, -1],  # M7: (b-d)
    ]
    
    # B coefficients: [e, f, g, h] for each Strassen term
    B_coeffs = [
        [1, 0, 0, 1],   # M1: (e+h)
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: (f-h)
        [-1, 0, 1, 0],  # M4: (g-e)
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: (e+f)
        [0, 0, 1, 1],   # M7: (g+h)
    ]
    
    # C coefficients: [C00, C01, C10, C11] for each Strassen term
    C_coeffs = [
        [1, 0, 0, 1],    # M1: C00+=1, C11+=1
        [0, 0, 1, -1],   # M2: C10+=1, C11+=-1
        [0, 1, 0, 1],    # M3: C01+=1, C11+=1
        [1, 0, 1, 0],    # M4: C00+=1, C10+=1
        [-1, 1, 0, 0],   # M5: C00+=-1, C01+=1
        [0, 0, 0, 1],    # M6: C11+=1
        [1, 0, 0, 0],    # M7: C00+=1
    ]
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    for b in range(num_blocks):
        for s in range(7):
            col = b * 7 + s
            
            # U: A coefficients (same for all blocks)
            for i in range(4):
                U[i][col] = A_coeffs[s][i]
            
            # V: B coefficients mapped to global indices
            # Block b uses B columns 2b and 2b+1
            # e = B[0][2b] -> index 0*p + 2b = 2b
            # f = B[0][2b+1] -> index 0*p + (2b+1) = 2b+1
            # g = B[1][2b] -> index 1*p + 2b = 14 + 2b
            # h = B[1][2b+1] -> index 1*p + (2b+1) = 14 + 2b + 1 = 15 + 2b
            idx_e = 2 * b
            idx_f = 2 * b + 1
            idx_g = 14 + 2 * b
            idx_h = 15 + 2 * b
            
            V[idx_e][col] = B_coeffs[s][0]
            V[idx_f][col] = B_coeffs[s][1]
            V[idx_g][col] = B_coeffs[s][2]
            V[idx_h][col] = B_coeffs[s][3]
            
            # W: C coefficients mapped to global indices
            # W[k*n+i] where k is column index of C, i is row index
            # C[0][2b]   -> k=2b,   i=0 -> W[2b*2+0] = W[4b]
            # C[0][2b+1] -> k=2b+1, i=0 -> W[(2b+1)*2+0] = W[4b+2]
            # C[1][2b]   -> k=2b,   i=1 -> W[2b*2+1] = W[4b+1]
            # C[1][2b+1] -> k=2b+1, i=1 -> W[(2b+1)*2+1] = W[4b+3]
            W[4 * b][col] = C_coeffs[s][0]       # C00
            W[4 * b + 2][col] = C_coeffs[s][1]   # C01
            W[4 * b + 1][col] = C_coeffs[s][2]   # C10
            W[4 * b + 3][col] = C_coeffs[s][3]   # C11
    
    return {"U": U, "V": V, "W": W}
