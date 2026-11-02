import time

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 blocks * 7 factors per block
    rank = 8 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # M1 = A00 (B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = (A00 + A01) B11
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A11) (B00 + B11)
    # M6 = (A01 - A11) (B10 + B11)
    # M7 = (A10 - A00) (B00 + B01)
    
    # Coefficients for U (A), V (B), W (C)
    # U[r][i][j], V[r][j][k], W[r][k][i]
    
    # We will define the 7 factors as lists of (i, j, coeff), (j, k, coeff), (k, i, coeff)
    # Note: W index is k*n + i, so W[k][i]
    
    factors = []
    
    # Factor 1
    # M1 = A00 (B00 + B11)
    u1 = {(0,0): 1}
    v1 = {(0,0): 1, (0,1): 1}
    # C01 = M1 + ... -> W1[1][0] = 1 (C10? No C01 is i=0, k=1. W index k*n+i = 1*2+0=2. W[1][0] is row 1 col 0? No. W is p x n. Row k, Col i.
    # C01 corresponds to i=0, k=1. W index = 1*2 + 0 = 2.
    # C11 corresponds to i=1, k=1. W index = 1*2 + 1 = 3.
    # C11 = ... - M1 ... -> W1[1][1] = -1.
    w1 = {(1,0): 1, (1,1): -1} # (k, i)
    factors.append((u1, v1, w1))
    
    # Factor 2
    # M2 = (A10 + A11) B00
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    # C10 = M2 + ... -> W2[0][1] = 1 (i=1, k=0. W index 0*2+1=1).
    # C11 = ... + M2 ... -> W2[1][1] = 1.
    w2 = {(0,1): 1, (1,1): 1}
    factors.append((u2, v2, w2))
    
    # Factor 3
    # M3 = (A00 + A01) B11
    u3 = {(0,0): 1, (0,1): 1}
    v3 = {(1,1): 1}
    # C01 = M1 + M3 -> W3[1][0] = 1.
    # C00 = ... - M3 ... -> W3[0][0] = -1.
    w3 = {(1,0): 1, (0,0): -1}
    factors.append((u3, v3, w3))
    
    # Factor 4
    # M4 = A11 (B10 - B00)
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    # C00 = ... + M4 ... -> W4[0][0] = 1.
    # C10 = M2 + M4 -> W4[0][1] = 1.
    w4 = {(0,0): 1, (0,1): 1}
    factors.append((u4, v4, w4))
    
    # Factor 5
    # M5 = (A00 + A11) (B00 + B11)
    u5 = {(0,0): 1, (1,1): 1}
    v5 = {(0,0): 1, (1,1): 1}
    # C00 = M5 + ... -> W5[0][0] = 1.
    # C11 = M5 + ... -> W5[1][1] = 1.
    w5 = {(0,0): 1, (1,1): 1}
    factors.append((u5, v5, w5))
    
    # Factor 6
    # M6 = (A01 - A11) (B10 + B11)
    u6 = {(0,1): 1, (1,1): -1}
    v6 = {(1,0): 1, (1,1): 1}
    # C11 = ... + M6 ... -> W6[1][1] = 1.
    w6 = {(1,1): 1}
    factors.append((u6, v6, w6))
    
    # Factor 7
    # M7 = (A10 - A00) (B00 + B01)
    u7 = {(1,0): 1, (0,0): -1}
    v7 = {(0,0): 1, (0,1): 1}
    # C00 = ... + M7 ... -> W7[0][0] = 1.
    w7 = {(0,0): 1}
    factors.append((u7, v7, w7))
    
    # Fill U, V, W for 8 blocks
    for b in range(8):
        for r_local in range(7):
            r_global = b * 7 + r_local
            u_dict, v_dict, w_dict = factors[r_local]
            
            # Fill U
            for (i, j), coeff in u_dict.items():
                U[i * m + j][r_global] = coeff
                
            # Fill V
            for (j, k_local), coeff in v_dict.items():
                k_global = b * 2 + k_local
                V[j * p + k_global][r_global] = coeff
                
            # Fill W
            for (k_local, i), coeff in w_dict.items():
                k_global = b * 2 + k_local
                W[k_global * n + i][r_global] = coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
