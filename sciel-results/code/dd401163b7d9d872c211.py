import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = 3, 3, 5
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: i in [0,1], j in [0,1]
    # Indices for B: j in [0,1], k in [0,1]
    # Indices for C: i in [0,1], k in [0,1]
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    def add_strassen_2x2x2(i_offset, j_offset, k_offset):
        nonlocal col
        # Strassen factors
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B01)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11 (with signs)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 + M3 - M2 - M6
        
        # We need to map these to U, V, W factors.
        # U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
        
        # M1: U: A00, A11. V: B00, B11.
        # U_{0,0,r} += 1, U_{1,1,r} += 1
        # V_{0,0,r} += 1, V_{1,1,r} += 1
        # W contributions:
        # C00 gets +M1. W_{0,0,r} += 1
        # C11 gets +M1. W_{1,1,r} += 1
        U[i_offset * m + j_offset][col] = 1
        U[(i_offset+1) * m + (j_offset+1)][col] = 1
        V[j_offset * p + k_offset][col] = 1
        V[(j_offset+1) * p + (k_offset+1)][col] = 1
        W[k_offset * n + i_offset][col] = 1
        W[(k_offset+1) * n + (i_offset+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        # U: A10, A11. V: B00.
        # C10 gets +M2. W_{0,1,r} += 1
        U[i_offset * m + j_offset][col] = 0 # A00 not involved
        U[(i_offset+1) * m + j_offset][col] = 1 # A10
        U[(i_offset+1) * m + (j_offset+1)][col] = 1 # A11
        V[j_offset * p + k_offset][col] = 1 # B00
        W[k_offset * n + (i_offset+1)][col] = 1 # C10
        col += 1
        
        # M3: A00(B11-B01)
        # U: A00. V: B11, B01.
        # C01 gets +M3. W_{1,0,r} += 1
        U[i_offset * m + j_offset][col] = 1 # A00
        V[(j_offset+1) * p + (k_offset+1)][col] = 1 # B11
        V[(j_offset+1) * p + k_offset][col] = -1 # B01
        W[(k_offset+1) * n + i_offset][col] = 1 # C01
        col += 1
        
        # M4: A11(B10-B00)
        # U: A11. V: B10, B00.
        # C00 gets +M4. W_{0,0,r} += 1
        # C10 gets +M4. W_{0,1,r} += 1
        U[(i_offset+1) * m + (j_offset+1)][col] = 1 # A11
        V[(j_offset+1) * p + k_offset][col] = 1 # B10
        V[j_offset * p + k_offset][col] = -1 # B00
        W[k_offset * n + i_offset][col] = 1 # C00
        W[k_offset * n + (i_offset+1)][col] = 1 # C10
        col += 1
        
        # M5: (A00+A01)B11
        # U: A00, A01. V: B11.
        # C01 gets +M5. W_{1,0,r} += 1
        # C00 gets -M5. W_{0,0,r} += -1
        U[i_offset * m + j_offset][col] = 1 # A00
        U[i_offset * m + (j_offset+1)][col] = 1 # A01
        V[(j_offset+1) * p + (k_offset+1)][col] = 1 # B11
        W[(k_offset+1) * n + i_offset][col] = 1 # C01
        W[k_offset * n + i_offset][col] = -1 # C00
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        # U: A10, A00. V: B00, B01.
        # C11 gets -M6. W_{1,1,r} += -1
        U[(i_offset+1) * m + j_offset][col] = 1 # A10
        U[i_offset * m + j_offset][col] = -1 # A00
        V[j_offset * p + k_offset][col] = 1 # B00
        V[j_offset * p + (k_offset+1)][col] = 1 # B01
        W[(k_offset+1) * n + (i_offset+1)][col] = -1 # C11
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        # U: A01, A11. V: B10, B11.
        # C00 gets -M7. W_{0,0,r} += -1
        U[i_offset * m + (j_offset+1)][col] = 1 # A01
        U[(i_offset+1) * m + (j_offset+1)][col] = -1 # A11
        V[(j_offset+1) * p + k_offset][col] = 1 # B10
        V[(j_offset+1) * p + (k_offset+1)][col] = 1 # B11
        W[k_offset * n + i_offset][col] = -1 # C00
        col += 1
        
    # T1: 2x2x5
    # k in 0,1 -> Strassen
    add_strassen_2x2x2(0, 0, 0)
    # k in 2,3 -> Strassen
    add_strassen_2x2x2(0, 0, 2)
    # k in 4 -> Schoolbook 2x2x1
    # A00 B00, A01 B10, A10 B01, A11 B11
    # U: A00, V: B00, W: C00
    U[0 * m + 0][col] = 1; V[0 * p + 4][col] = 1; W[4 * n + 0][col] = 1; col += 1
    # U: A01, V: B10, W: C01
    U[0 * m + 1][col] = 1; V[1 * p + 4][col] = 1; W[4 * n + 0][col] = 1; col += 1
    # U: A10, V: B01, W: C10
    U[1 * m + 0][col] = 1; V[0 * p + 4][col] = 1; W[4 * n + 1][col] = 1; col += 1
    # U: A11, V: B11, W: C11
    U[1 * m + 1][col] = 1; V[1 * p + 4][col] = 1; W[4 * n + 1][col] = 1; col += 1
    
    # T2: 2x1x5 (i in 0,1, j=2, k in 0..4)
    # A_i2 B_2k
    for i in range(2):
        for k in range(5):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # T3: 1x2x5 (i=2, j in 0,1, k in 0..4)
    # A_2j B_jk
    for j in range(2):
        for k in range(5):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    # T4: 1x1x5 (i=2, j=2, k in 0..4)
    # A_22 B_2k
    for k in range(5):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
