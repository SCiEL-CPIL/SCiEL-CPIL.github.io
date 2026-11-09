import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=4
    # A is 3x3, B is 3x4, C is 3x4
    
    # Indices for A: 0..8 (row-major 3x3)
    # Indices for B: 0..11 (row-major 3x4)
    # Indices for C: 0..11 (row-major 3x4)
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:4]
    # A indices: 0,1,3,4 (rows 0,1 cols 0,1)
    # B indices: 0..7 (rows 0,1 cols 0..3)
    # C indices: 0..7 (rows 0,1 cols 0..3)
    
    # Use 2x2x2 Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # And 2x2x2 Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    
    # Strassen 2x2x2 factors
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 - P4 + P5 - P7
    # C12 = P3 + P7
    # C21 = P2 + P4
    # C22 = P1 - P3 + P6 + P7
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
        nonlocal col
        # a_rows, a_cols are lists of global indices for A
        # b_rows, b_cols are lists of global indices for B
        # c_rows, c_cols are lists of global indices for C
        
        # P1: (a+d)(e+h)
        U[a_rows[0]*m + a_cols[0]][col] = 1
        U[a_rows[1]*m + a_cols[1]][col] = 1
        V[b_rows[0]*p + b_cols[0]][col] = 1
        V[b_rows[1]*p + b_cols[1]][col] = 1
        W[c_rows[0]*n + c_cols[0]][col] = 1
        W[c_rows[0]*n + c_cols[1]][col] = 1
        W[c_rows[1]*n + c_cols[0]][col] = 1
        W[c_rows[1]*n + c_cols[1]][col] = 1
        col += 1
        
        # P2: (c+d)e
        U[a_rows[1]*m + a_cols[0]][col] = 1
        U[a_rows[1]*m + a_cols[1]][col] = 1
        V[b_rows[0]*p + b_cols[0]][col] = 1
        W[c_rows[1]*n + c_cols[0]][col] = 1
        col += 1
        
        # P3: a(f-h)
        U[a_rows[0]*m + a_cols[0]][col] = 1
        V[b_rows[0]*p + b_cols[1]][col] = 1
        V[b_rows[1]*p + b_cols[1]][col] = -1
        W[c_rows[0]*n + c_cols[1]][col] = 1
        col += 1
        
        # P4: d(g-e)
        U[a_rows[1]*m + a_cols[1]][col] = 1
        V[b_rows[1]*p + b_cols[0]][col] = 1
        V[b_rows[0]*p + b_cols[0]][col] = -1
        W[c_rows[1]*n + c_cols[0]][col] = -1
        W[c_rows[0]*n + c_cols[0]][col] = 1
        col += 1
        
        # P5: (a+b)h
        U[a_rows[0]*m + a_cols[0]][col] = 1
        U[a_rows[0]*m + a_cols[1]][col] = 1
        V[b_rows[1]*p + b_cols[1]][col] = 1
        W[c_rows[1]*n + c_cols[1]][col] = 1
        col += 1
        
        # P6: (c-a)(e+f)
        U[a_rows[1]*m + a_cols[0]][col] = 1
        U[a_rows[0]*m + a_cols[0]][col] = -1
        V[b_rows[0]*p + b_cols[0]][col] = 1
        V[b_rows[0]*p + b_cols[1]][col] = 1
        W[c_rows[1]*n + c_cols[1]][col] = 1
        col += 1
        
        # P7: (b-d)(g+h)
        U[a_rows[0]*m + a_cols[1]][col] = 1
        U[a_rows[1]*m + a_cols[1]][col] = -1
        V[b_rows[1]*p + b_cols[0]][col] = 1
        V[b_rows[1]*p + b_cols[1]][col] = 1
        W[c_rows[0]*n + c_cols[1]][col] = 1
        W[c_rows[1]*n + c_cols[1]][col] = 1
        col += 1
        
    # A[0:2, 0:2] indices:
    a00 = 0*3 + 0
    a01 = 0*3 + 1
    a10 = 1*3 + 0
    a11 = 1*3 + 1
    
    # B[0:2, 0:2] indices:
    b00 = 0*4 + 0
    b01 = 0*4 + 1
    b10 = 1*4 + 0
    b11 = 1*4 + 1
    
    # C[0:2, 0:2] indices:
    c00 = 0*3 + 0
    c01 = 0*3 + 1
    c10 = 1*3 + 0
    c11 = 1*3 + 1
    
    # Strassen for C[0:2, 0:2]
    add_strassen([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0)
    
    # B[0:2, 2:4] indices:
    b02 = 0*4 + 2
    b03 = 0*4 + 3
    b12 = 1*4 + 2
    b13 = 1*4 + 3
    
    # C[0:2, 2:4] indices:
    c02 = 0*3 + 2
    c03 = 0*3 + 3
    c12 = 1*3 + 2
    c13 = 1*3 + 3
    
    # Strassen for C[0:2, 2:4]
    add_strassen([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 0)
    
    # Block 2: A[0:2, 2] * B[2, 0:4]
    # A indices: 2, 5 (col 2, rows 0,1)
    # B indices: 8,9,10,11 (row 2, cols 0..3)
    # C indices: 0..7 (rows 0,1 cols 0..3)
    # Rank 1
    U[2][col] = 1
    U[5][col] = 1
    V[8][col] = 1
    V[9][col] = 1
    V[10][col] = 1
    V[11][col] = 1
    W[0][col] = 1
    W[1][col] = 1
    W[2][col] = 1
    W[3][col] = 1
    W[4][col] = 1
    W[5][col] = 1
    W[6][col] = 1
    W[7][col] = 1
    col += 1
    
    # Block 3: A[2, 0:2] * B[0:2, 0:4]
    # A indices: 6, 7 (row 2, cols 0,1)
    # B indices: 0..7
    # C indices: 8..11 (row 2, cols 0..3)
    # Rank 8 (Schoolbook)
    for j in range(2):
        for k in range(4):
            U[6+j][col] = 1
            V[j*4+k][col] = 1
            W[8+k][col] = 1
            col += 1
            
    # Block 4: A[2, 2] * B[2, 0:4]
    # A index: 8
    # B indices: 8..11
    # C indices: 8..11
    # Rank 1
    U[8][col] = 1
    V[8][col] = 1
    V[9][col] = 1
    V[10][col] = 1
    V[11][col] = 1
    W[8][col] = 1
    W[9][col] = 1
    W[10][col] = 1
    W[11][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
