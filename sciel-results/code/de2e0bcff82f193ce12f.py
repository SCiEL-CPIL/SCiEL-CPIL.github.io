import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M1..M7
    # M1: (X00+X11)(Y00+Y11)
    # M2: (X10+X11)Y00
    # M3: X00(Y10-Y11)
    # M4: X11(Y01-Y00)
    # M5: (X00+X01)Y11
    # M6: (X10-X00)(Y00+Y10)
    # M7: (X01-X11)(Y11+Y01)
    
    # Coefficients for X part (2x2) and Y part (2x2)
    # x_coeffs[t][u][v], y_coeffs[t][u][v]
    x_coeffs = [
        [[1, 0], [0, 1]], # M1 X
        [[0, 0], [1, 1]], # M2 X
        [[1, 0], [0, 0]], # M3 X
        [[0, 0], [0, 1]], # M4 X
        [[1, 1], [0, 0]], # M5 X
        [[-1, 0], [1, 0]],# M6 X
        [[0, 1], [0, -1]] # M7 X
    ]
    y_coeffs = [
        [[1, 0], [0, 1]], # M1 Y
        [[1, 0], [0, 0]], # M2 Y
        [[0, -1], [0, 1]],# M3 Y (Y10-Y11 -> Y[1,0]-Y[1,1])
        [[-1, 1], [0, 0]],# M4 Y (Y01-Y00 -> Y[0,1]-Y[0,0])
        [[0, 0], [0, 1]], # M5 Y
        [[1, 1], [0, 0]], # M6 Y (Y00+Y10 -> Y[0,0]+Y[1,0])
        [[0, 1], [1, 0]]  # M7 Y (Y11+Y01 -> Y[1,1]+Y[0,1])
    ]
    
    # Reconstruction coefficients for Z
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M2 + M3
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M5
    # z_coeffs[u][v][t]
    z_coeffs = [
        [
            [1, 0, 0, 1, -1, 0, 1], # Z00
            [0, 1, 1, 0, 0, 0, 0]   # Z01
        ],
        [
            [0, 0, 0, 0, 0, 0, 0],  # Z10 (placeholder, will fill)
            [0, 0, 0, 0, 0, 0, 0]   # Z11
        ]
    ]
    # Fix Z10, Z11
    z_coeffs[1][0] = [0, 1, 0, 1, 0, 0, 0] # M2+M4
    z_coeffs[1][1] = [1, -1, 1, 0, 1, 0, 0] # M1-M2+M3+M5

    col = 0
    for s in range(2):
        for k in range(3):
            for t in range(7):
                # Process term t for instance (s, k)
                # s=0: A1 (cols 0,1), B rows 0,1
                # s=1: A2 (cols 2,3), B rows 2,3
                
                # Fill U
                for u in range(2):
                    for v in range(2):
                        coeff = x_coeffs[t][u][v]
                        if coeff != 0:
                            # A index: row u, col (s*2 + v)
                            a_idx = u * m + (s * 2 + v)
                            U[a_idx][col] += coeff
                            
                # Fill V
                for u in range(2):
                    for v in range(2):
                        coeff = y_coeffs[t][u][v]
                        if coeff != 0:
                            # B index: row (s*2 + u), col (k*2 + v)
                            b_idx = (s * 2 + u) * p + (k * 2 + v)
                            V[b_idx][col] += coeff
                            
                # Fill W
                for u in range(2):
                    for v in range(2):
                        coeff = z_coeffs[u][v][t]
                        if coeff != 0:
                            # C index: row u, col (k*2 + v)
                            # W index: k*n + i -> col * 2 + row
                            c_idx = (k * 2 + v) * n + u
                            W[c_idx][col] += coeff
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
