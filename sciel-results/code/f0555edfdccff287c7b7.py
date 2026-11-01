import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: i in 0..1, j in 0..2 -> idx = i*3 + j
    # V: j in 0..2, k in 0..2 -> idx = j*3 + k
    # W: k in 0..2, i in 0..1 -> idx = k*2 + i
    
    col = 0
    
    # Part 1: M_{2,1,3} (A col 0, B row 0)
    # C_{i,k} += A_{i,0} * B_{0,k}
    for i in range(2):
        for k in range(3):
            U[i * 3 + 0][col] = 1
            V[0 * 3 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    # col is now 6
    
    # Part 2: M_{2,2,3} (A cols 1,2; B rows 1,2)
    # Sub-matrix A_sub:
    # A01 = A[0,1] (U idx 1)
    # A02 = A[0,2] (U idx 2)
    # A11 = A[1,1] (U idx 4)
    # A12 = A[1,2] (U idx 5)
    
    # Sub-matrix B_sub:
    # B10 = B[1,0] (V idx 3)
    # B11 = B[1,1] (V idx 4)
    # B12 = B[1,2] (V idx 5)
    # B20 = B[2,0] (V idx 6)
    # B21 = B[2,1] (V idx 7)
    # B22 = B[2,2] (V idx 8)
    
    # Output C_sub:
    # C00 = C[0,0] (W idx 0)
    # C01 = C[0,1] (W idx 2)
    # C02 = C[0,2] (W idx 4)
    # C10 = C[1,0] (W idx 1)
    # C11 = C[1,1] (W idx 3)
    # C12 = C[1,2] (W idx 5)
    
    # Strassen for C00, C01, C10, C11 using B10, B11, B20, B21
    # m1 = (A01 + A12) * (B10 + B20)
    U[1][col] = 1; U[5][col] = 1
    V[3][col] = 1; V[6][col] = 1
    W[0][col] = 1; W[3][col] = 1; W[2][col] = 1; W[1][col] = -1 # C00, C11, C01, C10? No, need to map correctly later.
    # Wait, Strassen outputs are linear combinations.
    # It's easier to define the multiplications and then add to W.
    
    # Let's define the 7 Strassen mults explicitly.
    # m1
    U[1][col] = 1; U[5][col] = 1
    V[3][col] = 1; V[6][col] = 1
    # Contributions:
    # C00 += m1
    # C11 += m1
    W[0][col] = 1
    W[3][col] = 1
    col += 1
    
    # m2 = (A11 + A12) * B10
    U[4][col] = 1; U[5][col] = 1
    V[3][col] = 1
    # Contributions:
    # C10 += m2
    # C11 -= m2
    W[1][col] = 1
    W[3][col] = -1
    col += 1
    
    # m3 = A01 * (B11 - B20)
    U[1][col] = 1
    V[4][col] = 1; V[6][col] = -1
    # Contributions:
    # C01 += m3
    # C11 += m3
    W[2][col] = 1
    W[3][col] = 1
    col += 1
    
    # m4 = A12 * (B21 - B10)
    U[5][col] = 1
    V[7][col] = 1; V[3][col] = -1
    # Contributions:
    # C00 += m4
    # C10 += m4
    W[0][col] = 1
    W[1][col] = 1
    col += 1
    
    # m5 = (A01 + A02) * B21
    U[1][col] = 1; U[2][col] = 1
    V[7][col] = 1
    # Contributions:
    # C00 -= m5
    # C01 += m5
    W[0][col] = -1
    W[2][col] = 1
    col += 1
    
    # m6 = (A11 - A01) * (B10 + B20)
    U[4][col] = 1; U[1][col] = -1
    V[3][col] = 1; V[6][col] = 1
    # Contributions:
    # C11 += m6
    W[3][col] = 1
    col += 1
    
    # m7 = (A02 - A12) * (B21 + B10)
    U[2][col] = 1; U[5][col] = -1
    V[7][col] = 1; V[3][col] = 1
    # Contributions:
    # C00 += m7
    W[0][col] = 1
    col += 1
    
    # Now for C02 and C12 using B12, B22
    # C02 = A01*B12 + A02*B22
    # C12 = A11*B12 + A12*B22
    
    # n1 = A01 * B12
    U[1][col] = 1
    V[5][col] = 1
    W[4][col] = 1 # C02
    col += 1
    
    # n2 = A02 * B22
    U[2][col] = 1
    V[8][col] = 1
    W[4][col] = 1 # C02
    col += 1
    
    # n3 = A11 * B12
    U[4][col] = 1
    V[5][col] = 1
    W[5][col] = 1 # C12
    col += 1
    
    # n4 = A12 * B22
    U[5][col] = 1
    V[8][col] = 1
    W[5][col] = 1 # C12
    col += 1
    
    return {"U": U, "V": V, "W": W}
