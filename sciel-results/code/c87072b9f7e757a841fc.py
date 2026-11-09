"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen tiling: 6x6 into 3x3 blocks of 2x2
    # Rank = 27 * 7 = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # P1 = (X00+X11)(Y00+Y11)
    # P2 = (X10+X11)Y00
    # P3 = X00(Y01-Y11)
    # P4 = X11(Y10-Y00)
    # P5 = (X00+X01)Y11
    # P6 = (X10-X00)(Y00+Y01)
    # P7 = (X01-X11)(Y10+Y11)
    
    # Z00 = P1 + P4 - P5 + P7
    # Z01 = P3 + P5
    # Z10 = P2 + P4
    # Z11 = P1 - P2 + P3 + P6
    
    # Coefficients: { (row, col): coeff }
    # A coeffs, B coeffs, C coeffs
    strassen_coeffs = [
        # s=0: P1
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        # s=1: P2
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1}
        },
        # s=2: P3
        {
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        # s=3: P4
        {
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        # s=4: P5
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        # s=5: P6
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        # s=6: P7
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]
    
    r = 0
    for I in range(3):
        for K in range(3):
            for J in range(3):
                for s in range(7):
                    coeffs = strassen_coeffs[s]
                    
                    # Update U
                    for (ar, ac), val in coeffs['A'].items():
                        i = 2 * I + ar
                        j = 2 * K + ac
                        U[i * m + j][r] = val
                        
                    # Update V
                    for (br, bc), val in coeffs['B'].items():
                        j = 2 * K + br
                        k = 2 * J + bc
                        V[j * p + k][r] = val
                        
                    # Update W
                    for (cr, cc), val in coeffs['C'].items():
                        i = 2 * I + cr
                        k = 2 * J + cc
                        W[k * n + i][r] = val
                        
                    r += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
