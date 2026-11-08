"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen coefficients for U (A part) and V (B part)
    # X is 2x2, Y is 2x2
    # t=0: M1 = (X00+X11)(Y00+Y11)
    # t=1: M2 = (X10+X11)Y00
    # t=2: M3 = X00(Y01-Y11)
    # t=3: M4 = X11(Y00-Y01)
    # t=4: M5 = (X00+X01)Y11
    # t=5: M6 = (X10-X00)(Y00+Y01)
    # t=6: M7 = (X01-X11)(Y01+Y11)
    
    # U_coeffs[t] = [(local_i, local_j, coeff), ...]
    # V_coeffs[t] = [(local_j, local_k, coeff), ...]
    # W_coeffs[t] = [(local_i, local_k, coeff), ...]
    
    U_coeffs = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,1)],      # M2
        [(0,0,1)],               # M3
        [(1,1,1)],               # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (0,0,-1)],     # M6
        [(0,1,1), (1,1,-1)]      # M7
    ]
    
    V_coeffs = [
        [(0,0,1), (1,1,1)],      # M1
        [(0,0,1)],               # M2
        [(1,0,1), (1,1,-1)],     # M3
        [(0,0,1), (1,0,-1)],     # M4
        [(1,1,1)],               # M5
        [(0,0,1), (1,0,1)],      # M6
        [(1,0,1), (1,1,1)]       # M7
    ]
    
    # Reconstruction coefficients for W
    # Z00 = M1 - M4 + M5 - M7
    # Z01 = M3 + M4 + M7
    # Z10 = M2 + M3 - M5 - M6
    # Z11 = M1 - M2 + M6
    
    # W_coeffs[t] = [(local_i, local_k, coeff), ...]
    W_coeffs = [
        [(0,0,1), (1,1,-1)],     # M1 contributes to Z00(+), Z11(-)
        [(1,0,1), (1,1,-1)],     # M2 contributes to Z10(+), Z11(-)
        [(0,1,1), (1,0,1)],      # M3 contributes to Z01(+), Z10(+)
        [(0,0,-1), (0,1,1)],     # M4 contributes to Z00(-), Z01(+)
        [(0,0,1), (1,0,-1)],     # M5 contributes to Z00(+), Z10(-)
        [(1,0,-1), (1,1,1)],     # M6 contributes to Z10(-), Z11(+)
        [(0,0,-1), (0,1,1)]      # M7 contributes to Z00(-), Z01(+)
    ]
    
    for q in range(2):  # Part 0 (A1, B1) and Part 1 (A2, B2)
        for s in range(4):  # Block 0..3 (cols 0-1, 2-3, 4-5, 6-7)
            for t in range(7):  # Strassen term 0..6
                # Process U
                for (li, lj, coeff) in U_coeffs[t]:
                    # Global i is li (0 or 1)
                    # Global j is q*2 + lj
                    i_idx = li
                    j_idx = q * 2 + lj
                    U[i_idx * m + j_idx][col] = coeff
                
                # Process V
                for (lj, lk, coeff) in V_coeffs[t]:
                    # Global j is q*2 + lj
                    # Global k is s*2 + lk
                    j_idx = q * 2 + lj
                    k_idx = s * 2 + lk
                    V[j_idx * p + k_idx][col] = coeff
                
                # Process W
                for (li, lk, coeff) in W_coeffs[t]:
                    # Global i is li
                    # Global k is s*2 + lk
                    i_idx = li
                    k_idx = s * 2 + lk
                    W[k_idx * n + i_idx][col] = coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
