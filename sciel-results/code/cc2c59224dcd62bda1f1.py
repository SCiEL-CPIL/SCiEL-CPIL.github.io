import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    A_coeffs = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    
    B_coeffs = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [-1, 0, 1, 0],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    
    C_coeffs = [
        [1, 0, 0, 0], # M1
        [0, 0, 0, 0], # M2
        [0, 0, 0, 0], # M3
        [1, 0, 0, 0], # M4
        [-1, 0, 0, 0],# M5
        [0, 0, 0, 0], # M6
        [1, 0, 0, 0]  # M7
    ]
    C_coeffs[2][1] = 1
    C_coeffs[4][1] = 1
    C_coeffs[1][2] = 1
    C_coeffs[3][2] = 1
    C_coeffs[0][3] = 1
    C_coeffs[1][3] = -1
    C_coeffs[2][3] = 1
    C_coeffs[5][3] = 1
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(4):
        for type_idx in range(2):
            if type_idx == 0:
                j_offset_A = 0
                j_offset_B = 0
            else:
                j_offset_A = 2
                j_offset_B = 2
            k_offset = 2 * l
            
            for r in range(7):
                for a in range(2):
                    for b in range(2):
                        val = A_coeffs[r][a * 2 + b]
                        if val != 0:
                            U[a * 4 + (j_offset_A + b)][col] = val
                            
                for b in range(2):
                    for c in range(2):
                        val = B_coeffs[r][b * 2 + c]
                        if val != 0:
                            V[(j_offset_B + b) * 8 + (k_offset + c)][col] = val
                            
                for a in range(2):
                    for c in range(2):
                        val = C_coeffs[r][a * 2 + c]
                        if val != 0:
                            W[(k_offset + c) * 2 + a][col] = val
                            
                col += 1
                
    return {"U": U, "V": V, "W": W}
