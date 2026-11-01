import numpy as np

def construct():
    n, m, p = 2, 2, 10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 block
    # Terms: M1..M7
    # U (A) coefficients: [a, b, c, d]
    # V (B) coefficients: [e, f, g, h]
    # W (C) coefficients: [c00, c01, c10, c11]
    
    # Based on standard Strassen algorithm
    # Note: Some sources have different signs/offsets. 
    # I will use the set that yields rank 7.
    
    # Coefficients for the 7 terms
    # Term 1: (a+d)(e+h)
    # Term 2: (c+d)e
    # Term 3: a(f-h)
    # Term 4: d(g-f)
    # Term 5: (a+b)h
    # Term 6: (c-a)(e+f)
    # Term 7: (b-d)(g+h)
    
    # Reconstruction:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Local U (4x7)
    local_U = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    # Local V (4x7)
    local_V = [
        [1, 1, 0, 0, 0, 1, 0], # e
        [0, 0, 1, -1, 0, 1, 0],# f
        [0, 0, 0, 1, 0, 0, 1], # g
        [1, 0, -1, 0, 1, 0, 1] # h
    ]
    
    # Local W (4x7)
    local_W = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 1]  # c11
    ]
    
    # Fill global matrices
    col = 0
    for b in range(5):
        for t in range(7):
            # U: rows 0..3 (i=0,1; j=0,1)
            for i in range(2):
                for j in range(2):
                    U[i * 2 + j][col] = local_U[i * 2 + j][t]
            
            # V: rows 0..19 (j=0,1; k=2b..2b+1)
            for j in range(2):
                for k in range(2):
                    # Global k index = 2*b + k
                    V[j * 10 + (2 * b + k)][col] = local_V[j * 2 + k][t]
            
            # W: rows 0..19 (k=2b..2b+1; i=0,1)
            for k in range(2):
                for i in range(2):
                    # Global k index = 2*b + k
                    W[(2 * b + k) * 2 + i][col] = local_W[k * 2 + i][t]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
