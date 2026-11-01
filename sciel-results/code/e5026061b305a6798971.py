import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 17 decomposition for M_{2,3,3}
    # Split A into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    # Split B into B1 (2x3, rows 0,1) and B2 (1x3, row 2)
    # C = A1 B1 + A2 B2
    # A1 B1 is M_{2,2,3} with rank 11 (7 from Strassen on first 2 cols, 4 from trivial on last col)
    # A2 B2 is M_{2,1,3} with rank 6 (trivial)
    
    rank = 17
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Indices in U (A): 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # Indices in V (B): 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2), 6:(2,0), 7:(2,1), 8:(2,2)
    # Indices in W (C): 0:(0,0), 1:(0,1), 2:(0,2), 3:(1,0), 4:(1,1), 5:(1,2)
    # Note: W index is k*n + i. 
    # C[0,0] -> 0*2+0=0
    # C[0,1] -> 1*2+0=2
    # C[0,2] -> 2*2+0=4
    # C[1,0] -> 0*2+1=1
    # C[1,1] -> 1*2+1=3
    # C[1,2] -> 2*2+1=5
    
    # --- Part 1: A1 B1 (M_{2,2,3}) ---
    # Strassen terms for first 2 columns of C (using B cols 0,1)
    # A1 indices: 0,1,3,4 (a00, a01, a10, a11)
    # B1 indices for cols 0,1: 0,3 (b00, b10) and 1,4 (b01, b11)
    
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1  # C00
    W[5, 0] = 1  # C11 (Wait, C11 is index 3? No. C[i,k]. i=1, k=1 -> 1*2+1=3. C11 is index 3. C00 is 0. C11 is 3.)
    # Correction on W indices:
    # C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5
    
    W[0, 0] = 1  # C00
    W[3, 0] = 1  # C11
    
    # M2 = (a10+a11)b00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # C10
    W[3, 1] = -1 # C11
    
    # M3 = a00(b01-b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1  # C01
    W[3, 2] = 1  # C11
    
    # M4 = a11(b10-b00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C00
    W[1, 3] = 1  # C10
    
    # M5 = (a00+a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1 # C00
    W[2, 4] = 1  # C01
    
    # M6 = (a10-a00)(b00+b01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11
    
    # M7 = (a01-a11)(b10+b11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1  # C00
    W[3, 6] = 1  # C11
    
    # Trivial terms for 3rd column of C (using B col 2: b02, b12)
    # N1 = a00 b02
    U[0, 7] = 1
    V[2, 7] = 1
    W[4, 7] = 1  # C02
    
    # N2 = a01 b12
    U[1, 8] = 1
    V[5, 8] = 1
    W[4, 8] = 1  # C02
    
    # N3 = a10 b02
    U[3, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1  # C12
    
    # N4 = a11 b12
    U[4, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1  # C12
    
    # --- Part 2: A2 B2 (M_{2,1,3}) ---
    # A2 indices: 2 (a02), 5 (a12)
    # B2 indices: 6 (b20), 7 (b21), 8 (b22)
    
    # P1 = a02 b20
    U[2, 11] = 1
    V[6, 11] = 1
    W[0, 11] = 1  # C00
    
    # P2 = a02 b21
    U[2, 12] = 1
    V[7, 12] = 1
    W[2, 12] = 1  # C01
    
    # P3 = a02 b22
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1  # C02
    
    # P4 = a12 b20
    U[5, 14] = 1
    V[6, 14] = 1
    W[1, 14] = 1  # C10
    
    # P5 = a12 b21
    U[5, 15] = 1
    V[7, 15] = 1
    W[3, 15] = 1  # C11
    
    # P6 = a12 b22
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1  # C12
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
