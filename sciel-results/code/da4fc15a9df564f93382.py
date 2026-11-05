import time
import sys

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 189
    rank = 189
    
    # Initialize matrices with 0
    # U: (n*m, rank) = (36, 189)
    # V: (m*p, rank) = (36, 189)
    # W: (p*n, rank) = (36, 189)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (X) and B (Y)
    # s=0..6
    # A_part[s] is a list of (u, v, coeff)
    # B_part[s] is a list of (v, w, coeff)
    # Note: u, v, w are local indices 0,1
    
    A_parts = [
        [(0, 0, 1), (1, 1, 1)],       # M1: X00 + X11
        [(1, 0, 1), (1, 1, 1)],       # M2: X10 + X11
        [(0, 0, 1)],                  # M3: X00
        [(1, 1, 1)],                  # M4: X11
        [(0, 0, 1), (0, 1, 1)],       # M5: X00 + X01
        [(1, 0, 1), (0, 0, -1)],      # M6: X10 - X00
        [(0, 1, 1), (1, 1, -1)]       # M7: X01 - X11
    ]
    
    B_parts = [
        [(0, 0, 1), (1, 1, 1)],       # M1: Y00 + Y11
        [(0, 0, 1)],                  # M2: Y00
        [(1, 0, 1), (1, 1, -1)],      # M3: Y10 - Y11
        [(0, 1, 1), (0, 0, -1)],      # M4: Y01 - Y00
        [(1, 1, 1)],                  # M5: Y11
        [(0, 0, 1), (1, 0, 1)],       # M6: Y00 + Y10
        [(1, 1, 1), (0, 1, 1)]        # M7: Y11 + Y01
    ]
    
    # W coefficients: W_coeff[s][u][w]
    # P00 = M1 + M4 - M5 + M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M2 + M3 + M6
    
    W_coeffs = [
        # M1
        [[1, 0], [0, 1]],
        # M2
        [[0, 0], [1, -1]],
        # M3
        [[0, 1], [0, 1]],
        # M4
        [[1, 0], [1, 0]],
        # M5
        [[-1, 1], [0, 0]],
        # M6
        [[0, 0], [0, 1]],
        # M7
        [[1, 0], [0, 0]]
    ]
    
    # Iterate over block indices
    # I: row block of A (0..2)
    # J: col block of A / row block of B (0..2)
    # K: col block of B (0..2)
    
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    # Calculate global rank index
                    r = ((I * 3 + J) * 3 + K) * 7 + s
                    
                    # Populate U
                    for (u, v, coeff) in A_parts[s]:
                        i = 2 * I + u
                        j = 2 * J + v
                        U[i * m + j][r] += coeff
                        
                    # Populate V
                    for (v, w, coeff) in B_parts[s]:
                        j = 2 * J + v
                        k = 2 * K + w
                        V[j * p + k][r] += coeff
                        
                    # Populate W
                    for u in range(2):
                        for w in range(2):
                            coeff = W_coeffs[s][u][w]
                            if coeff != 0:
                                i = 2 * I + u
                                k = 2 * K + w
                                W[k * n + i][r] += coeff
                                
    return {"U": U, "V": V, "W": W}

