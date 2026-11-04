import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 29: 4 * 7 (Strassen for 2x2x2 blocks) + 1 (for 2x1x8)
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # U coefficients (A part)
    # Indices: i, j. i in 0..1, j in 0..1
    U_strassen = [
        [1, 0, 1, 0, 1, 0, 0], # i=0, j=0 (A11)
        [0, 1, 0, 1, 1, 1, 1], # i=0, j=1 (A12) - Wait, M6 has A21-A11. A12 is in M5, M7.
        # Let's re-verify M6: (A21-A11)(B11+B12). A21 is i=1, j=0. A11 is i=0, j=0.
        # So U[1,0] has -1 for M6. U[0,0] has -1 for M6.
        # Let's list carefully.
    ]
    
    # Let's use a list of tuples (i, j, coeff) for U
    U_terms = [
        # c=0 (M1)
        (0, 0, 1), (1, 1, 1),
        # c=1 (M2)
        (1, 0, 1), (1, 1, 1),
        # c=2 (M3)
        (0, 0, 1),
        # c=3 (M4)
        (1, 1, 1),
        # c=4 (M5)
        (0, 0, 1), (0, 1, 1),
        # c=5 (M6)
        (1, 0, 1), (0, 0, -1),
        # c=6 (M7)
        (0, 1, 1), (1, 1, -1)
    ]
    
    V_terms = [
        # c=0 (M1)
        (0, 0, 1), (1, 1, 1),
        # c=1 (M2)
        (0, 0, 1),
        # c=2 (M3)
        (0, 1, 1), (1, 1, -1),
        # c=3 (M4)
        (1, 0, 1), (0, 0, -1),
        # c=4 (M5)
        (1, 1, 1),
        # c=5 (M6)
        (0, 0, 1), (0, 1, 1),
        # c=6 (M7)
        (1, 0, 1), (1, 1, 1)
    ]
    
    W_terms = [
        # c=0 (M1): C11+=1, C12+=0, C21+=-1, C22+=1
        (0, 0, 1), (1, 0, -1), (1, 1, 1),
        # c=1 (M2): C11+=0, C12+=0, C21+=1, C22+=-1
        (1, 0, 1), (1, 1, -1),
        # c=2 (M3): C11+=0, C12+=1, C21+=1, C22+=0
        (0, 1, 1), (1, 0, 1),
        # c=3 (M4): C11+=-1, C12+=1, C21+=0, C22+=0
        (0, 0, -1), (0, 1, 1),
        # c=4 (M5): C11+=1, C12+=0, C21+=0, C22+=1
        (0, 0, 1), (1, 1, 1),
        # c=5 (M6): C11+=0, C12+=0, C21+=1, C22+=-1
        (1, 0, 1), (1, 1, -1),
        # c=6 (M7): C11+=-1, C12+=1, C21+=0, C22+=0
        (0, 0, -1), (0, 1, 1)
    ]
    
    # Fill U, V, W for the 4 blocks of j in {0,1}
    for block in range(4):
        k_offset = block * 2
        for c in range(7):
            col = block * 7 + c
            for i, j, val in U_terms:
                if val != 0:
                    U[i * m + j][col] = val
            for j, k, val in V_terms:
                if val != 0:
                    V[j * p + (k_offset + k)][col] = val
            for k, i, val in W_terms:
                if val != 0:
                    W[(k_offset + k) * n + i][col] = val
    
    # Fill U, V, W for j=2
    col = 28
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
