"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 7 terms
    # Each term is (U_indices, U_coeffs, V_indices, V_coeffs, W_indices, W_coeffs)
    # Indices are relative to the 2x2 block
    # U: (i, j), V: (j, k), W: (k, i)
    
    for s in range(p // 2):
        base_v = 2 * s
        base_w = 4 * s
        # V indices:
        # B_00: 0*p + (2s) = 2s
        # B_01: 0*p + (2s+1) = 2s+1
        # B_10: 1*p + (2s) = 14+2s
        # B_11: 1*p + (2s+1) = 15+2s
        
        # W indices:
        # C_00: (2s)*2 + 0 = 4s
        # C_01: (2s+1)*2 + 0 = 4s+2
        # C_10: (2s)*2 + 1 = 4s+1
        # C_11: (2s+1)*2 + 1 = 4s+3
        
        for t in range(7):
            r = 7 * s + t
            if t == 0: # M1
                U[0][r] = 1; U[3][r] = 1
                V[base_v][r] = 1; V[14 + base_v + 1][r] = 1
                W[base_w][r] = 1; W[base_w + 3][r] = 1
            elif t == 1: # M2
                U[2][r] = 1; U[3][r] = 1
                V[base_v][r] = 1
                W[base_w + 1][r] = 1; W[base_w + 3][r] = -1
            elif t == 2: # M3
                U[0][r] = 1
                V[base_v + 1][r] = 1; V[14 + base_v + 1][r] = -1
                W[base_w + 2][r] = 1; W[base_w + 3][r] = 1
            elif t == 3: # M4
                U[3][r] = 1
                V[14 + base_v][r] = 1; V[base_v][r] = -1
                W[base_w][r] = 1; W[base_w + 1][r] = 1
            elif t == 4: # M5
                U[0][r] = 1; U[1][r] = 1
                V[14 + base_v + 1][r] = 1
                W[base_w][r] = -1; W[base_w + 2][r] = 1
            elif t == 5: # M6
                U[2][r] = 1; U[0][r] = -1
                V[base_v][r] = 1; V[base_v + 1][r] = 1
                W[base_w + 3][r] = 1
            elif t == 6: # M7
                U[1][r] = 1; U[3][r] = -1
                V[14 + base_v][r] = 1; V[14 + base_v + 1][r] = 1
                W[base_w][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
