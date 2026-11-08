"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Split m=4 into two groups of 2 (j1 in 0,1)
    # Split p=6 into three groups of 2 (k2 in 0,1,2)
    # Each block is 2x2x2 Strassen
    # Total rank = 2 * 3 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (u, v) and B (v, w)
    # u, v, w in {0, 1}
    # t in 1..7
    
    # U_coeffs[t][u][v]
    U_coeffs = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): 1},
        3: {(0,0): 1},
        4: {(1,1): 1},
        5: {(0,0): 1, (0,1): 1},
        6: {(1,0): 1, (0,0): -1},
        7: {(0,1): 1, (1,1): -1}
    }
    
    # V_coeffs[t][v][w]
    V_coeffs = {
        1: {(0,0): 1, (1,1): 1},
        2: {(0,0): 1},
        3: {(0,1): 1, (1,1): -1},
        4: {(1,0): 1, (0,0): -1},
        5: {(1,1): 1},
        6: {(0,0): 1, (0,1): 1},
        7: {(1,0): 1, (1,1): 1}
    }
    
    # W_coeffs[t][u][w]
    # C_00 = P1 + P4 - P5 + P7
    # C_01 = P3 + P5
    # C_10 = P2 + P4
    # C_11 = P1 - P2 + P3 + P6
    
    W_coeffs = {
        1: {(0,0): 1, (1,1): 1},
        2: {(1,0): 1, (1,1): -1},
        3: {(0,1): 1, (1,1): 1},
        4: {(0,0): 1, (1,0): 1},
        5: {(0,0): -1, (0,1): 1},
        6: {(1,1): 1},
        7: {(0,0): 1}
    }
    
    col = 0
    for j1 in range(2): # 2 blocks in m
        for k2 in range(3): # 3 blocks in p
            for t in range(1, 8): # 7 Strassen products
                # U
                for u in range(2):
                    for v in range(2):
                        coeff = U_coeffs[t].get((u, v), 0)
                        if coeff != 0:
                            i = u
                            j = 2 * j1 + v
                            U[i * m + j][col] += coeff
                            
                # V
                for v in range(2):
                    for w in range(2):
                        coeff = V_coeffs[t].get((v, w), 0)
                        if coeff != 0:
                            j = 2 * j1 + v
                            k = 2 * k2 + w
                            V[j * p + k][col] += coeff
                            
                # W
                for u in range(2):
                    for w in range(2):
                        coeff = W_coeffs[t].get((u, w), 0)
                        if coeff != 0:
                            i = u
                            k = 2 * k2 + w
                            W[k * n + i][col] += coeff
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
