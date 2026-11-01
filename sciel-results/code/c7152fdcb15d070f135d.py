"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x14 multiplication into 7 independent 2x2 multiplications
    # using Strassen's algorithm for each 2x2 block.
    # Strassen rank for 2x2 is 7.
    # Total rank = 7 (blocks) * 7 (Strassen) = 49.
    
    # Strassen decomposition coefficients for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: (i, j) -> i*m + j. i in {0,1}, j in {0,1}
    # V indices: (j, k_local) -> j*p + k_global. Here k_global = block * 2 + k_local
    # W indices: (k_global, i) -> k_global * n + i.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    # Define the 7 Strassen steps
    # Each step has coefficients for U (A), V (B), W (C)
    # u_coeff: dict mapping (i, j) to coeff
    # v_coeff: dict mapping (j, k_local) to coeff
    # w_coeff: dict mapping (k_local, i) to coeff
    
    steps = []
    
    # Step 0: M1 = (A00 + A11) * (B00 + B11)
    u0 = {(0,0): 1, (1,1): 1}
    v0 = {(0,0): 1, (1,1): 1}
    w0 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): -1}  # C00+=1, C01+=0? No.
    # Let's trace C contributions carefully.
    # M1 contributes to:
    # C00 += M1
    # C11 += M1
    # So w0 should be {(0,0): 1, (1,1): 1}
    w0 = {(0,0): 1, (1,1): 1}
    steps.append((u0, v0, w0))
    
    # Step 1: M2 = (A10 + A11) * B00
    u1 = {(1,0): 1, (1,1): 1}
    v1 = {(0,0): 1}
    w1 = {(1,0): 1, (1,1): -1} # C10 += M2, C11 -= M2
    steps.append((u1, v1, w1))
    
    # Step 2: M3 = A00 * (B10 - B11)
    u2 = {(0,0): 1}
    v2 = {(1,0): 1, (1,1): -1}
    w2 = {(0,1): 1, (1,0): 1} # C01 += M3, C10 += M3
    steps.append((u2, v2, w2))
    
    # Step 3: M4 = A11 * (B01 - B00)
    u3 = {(1,1): 1}
    v3 = {(0,1): 1, (0,0): -1}
    w3 = {(0,0): 1, (1,0): 1} # C00 += M4, C10 += M4
    steps.append((u3, v3, w3))
    
    # Step 4: M5 = (A00 + A01) * B11
    u4 = {(0,0): 1, (0,1): 1}
    v4 = {(1,1): 1}
    w4 = {(0,0): -1, (0,1): 1} # C00 -= M5, C01 += M5
    steps.append((u4, v4, w4))
    
    # Step 5: M6 = (A10 - A00) * (B00 + B10)
    u5 = {(1,0): 1, (0,0): -1}
    v5 = {(0,0): 1, (1,0): 1}
    w5 = {(1,1): 1} # C11 += M6
    steps.append((u5, v5, w5))
    
    # Step 6: M7 = (A01 - A11) * (B01 + B11)
    u6 = {(0,1): 1, (1,1): -1}
    v6 = {(0,1): 1, (1,1): 1}
    w6 = {(0,0): 1} # C00 += M7
    steps.append((u6, v6, w6))
    
    # Now populate U, V, W
    for block in range(7):
        for s in range(7):
            r = block * 7 + s
            u_dict, v_dict, w_dict = steps[s]
            
            # U
            for (i, j), coeff in u_dict.items():
                U[i * m + j][r] = coeff
            
            # V
            for (j, k_local), coeff in v_dict.items():
                k_global = block * 2 + k_local
                V[j * p + k_global][r] = coeff
            
            # W
            for (k_local, i), coeff in w_dict.items():
                k_global = block * 2 + k_local
                W[k_global * n + i][r] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
