"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition using two Strassen 2x2x2 multiplications
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (U)
    # Indices: 0: A00, 1: A01, 2: A10, 3: A11
    # 1: A00+A11
    # 2: A10+A11
    # 3: A00
    # 4: A11
    # 5: A00+A01
    # 6: A10-A00
    # 7: A01-A11
    
    U_coeffs = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [-1, 0, 1, 0],# 6
        [0, 1, 0, -1] # 7
    ]
    
    # Fill U for first 7 cols (Left block) and next 7 cols (Right block)
    for c in range(7):
        for i in range(4):
            U[i][c] = U_coeffs[c][i]
            U[i][c + 7] = U_coeffs[c][i]
            
    # Strassen coefficients for B_L (V, cols 0-6)
    # B_L indices: 0: B00, 1: B01, 4: B10, 5: B11
    # V indices: j*p+k
    # j=0: 0,1,2,3
    # j=1: 4,5,6,7
    # B00 is idx 0, B01 is idx 1, B10 is idx 4, B11 is idx 5
    
    V_left_coeffs = [
        [1, 0, 0, 0, 0, 1, 0, 0], # 1: B00+B11
        [1, 0, 0, 0, 0, 0, 0, 0], # 2: B00
        [0, 0, 0, 0, 1, -1, 0, 0],# 3: B10-B11
        [0, 0, 0, 0, 0, 1, 0, 0], # 4: B11
        [0, 0, 0, 0, 0, 1, 0, 0], # 5: B11
        [1, 0, 0, 0, 1, 0, 0, 0], # 6: B00+B10
        [0, 1, 0, 0, 0, 1, 0, 0]  # 7: B11+B01
    ]
    
    for c in range(7):
        for i in range(8):
            V[i][c] = V_left_coeffs[c][i]
            
    # Strassen coefficients for B_R (V, cols 7-13)
    # B_R indices: 2: B02, 3: B03, 6: B12, 7: B13
    
    V_right_coeffs = [
        [0, 0, 1, 0, 0, 0, 0, 1], # 1: B02+B13
        [0, 0, 1, 0, 0, 0, 0, 0], # 2: B02
        [0, 0, 0, 0, 0, 0, 1, -1],# 3: B12-B13
        [0, 0, 0, 0, 0, 0, 0, 1], # 4: B13
        [0, 0, 0, 0, 0, 0, 0, 1], # 5: B13
        [0, 0, 1, 0, 0, 0, 1, 0], # 6: B02+B12
        [0, 1, 0, 0, 0, 0, 0, 1]  # 7: B13+B03
    ]
    
    for c in range(7):
        for i in range(8):
            V[i][c + 7] = V_right_coeffs[c][i]
            
    # Strassen output reconstruction for C_L (W, cols 0-6)
    # C_L indices: 0: C00, 1: C10, 2: C01, 3: C11
    # W indices: k*n+i
    # k=0: 0,1
    # k=1: 2,3
    
    W_left_coeffs = [
        [1, 0, -1, 1, 0, 0, -1], # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 0, 0, 1, 0, 1],   # C01
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    for c in range(7):
        for i in range(4):
            W[i][c] = W_left_coeffs[i][c]
            
    # Strassen output reconstruction for C_R (W, cols 7-13)
    # C_R indices: 4: C02, 5: C12, 6: C03, 7: C13
    
    W_right_coeffs = [
        [1, 0, -1, 1, 0, 0, -1], # C02
        [0, 1, 0, 1, 0, 0, 0],   # C12
        [0, 0, 0, 0, 1, 0, 1],   # C03
        [1, -1, 1, 0, 0, 1, 0]   # C13
    ]
    
    for c in range(7):
        for i in range(4):
            W[i + 4][c + 7] = W_right_coeffs[i][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
