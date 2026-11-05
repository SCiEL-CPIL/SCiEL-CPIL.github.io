import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=8
    # U: 9 x R
    # V: 24 x R
    # W: 24 x R
    
    R = 71
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Part 1: M_3,3,3 (Rank 26)
    # B columns 0, 1, 2
    
    # Indices for A (3x3):
    # 00, 01, 02
    # 10, 11, 12
    # 20, 21, 22
    # U index = i*m + j
    
    # Indices for B (3x8):
    # j0, j1, j2 (for 3x3 part)
    # j3...j7 (for 3x5 part)
    # V index = j*p + k
    
    # Indices for C (8x3):
    # k0i0, k0i1, k0i2
    # k1i0, k1i1, k1i2
    # k2i0, k2i1, k2i2
    # k3i0 ... k7i2
    # W index = k*n + i
    
    col = 0
    
    # --- Strassen 7 terms for A22 * B22 ---
    # A22: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices 0, 1, 3, 4
    # B22: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices 0, 1, 8, 9 (k=0,1)
    # C22: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices 0, 1, 3, 4 (k=0,1)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, col] += 1; U[4, col] += 1
    V[0, col] += 1; V[9, col] += 1
    W[0, col] += 1; W[1, col] += 1; W[3, col] += 1; W[4, col] -= 1
    col += 1
    
    # M2 = (A10+A11)*B00
    U[3, col] += 1; U[4, col] += 1
    V[0, col] += 1
    W[3, col] += 1; W[4, col] -= 1
    col += 1
    
    # M3 = A00*(B01-B11)
    U[0, col] += 1
    V[1, col] += 1; V[9, col] -= 1
    W[1, col] += 1; W[4, col] -= 1
    col += 1
    
    # M4 = A11*(B10-B01)
    U[4, col] += 1
    V[8, col] += 1; V[1, col] -= 1
    W[0, col] += 1; W[3, col] += 1
    col += 1
    
    # M5 = (A00+A01)*B11
    U[0, col] += 1; U[1, col] += 1
    V[9, col] += 1
    W[0, col] -= 1; W[1, col] += 1
    col += 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[3, col] += 1; U[0, col] -= 1
    V[0, col] += 1; V[1, col] += 1
    W[3, col] += 1; W[4, col] += 1
    col += 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, col] += 1; U[4, col] -= 1
    V[8, col] += 1; V[9, col] += 1
    W[0, col] += 1; W[3, col] -= 1
    col += 1
    
    # --- Cross terms for M_3,3,3 ---
    # a = A[:,2] (2x1), e = B[2,:] (1x2) for top-right?
    # Partition:
    # A = [ A22, a ]
    #     [ bT,  c ]
    # a: A[0,2], A[1,2] -> U 2, 5
    # b: A[2,0], A[2,1] -> U 6, 7
    # c: A[2,2] -> U 8
    
    # B = [ B22, d ]
    #     [ eT,  f ]
    # d: B[0,2], B[1,2] -> V 2, 10
    # e: B[2,0], B[2,1] -> V 16, 17
    # f: B[2,2] -> V 18
    
    # C = [ C22, Ctr ]
    #     [ Cbl, Cbr ]
    # Ctr: C[0,2], C[1,2] -> W 2, 5
    # Cbl: C[2,0], C[2,1] -> W 6, 7
    # Cbr: C[2,2] -> W 8
    
    # 1. a eT (2x1 * 1x2) -> C22 top-right? No.
    # A[0:2, 2] * B[2, 0:2] contributes to C[0:2, 0:2] (top-left)
    # Terms: a_x e_y -> C[xy]
    # x in 0,1; y in 0,1.
    # a0=A02 (U2), a1=A12 (U5)
    # e0=B20 (V16), e1=B21 (V17)
    # C00 (W0), C01 (W1), C10 (W3), C11 (W4)
    
    # a0 e0 -> C00
    U[2, col] += 1; V[16, col] += 1; W[0, col] += 1
    col += 1
    # a0 e1 -> C01
    U[2, col] += 1; V[17, col] += 1; W[1, col] += 1
    col += 1
    # a1 e0 -> C10
    U[5, col] += 1; V[16, col] += 1; W[3, col] += 1
    col += 1
    # a1 e1 -> C11
    U[5, col] += 1; V[17, col] += 1; W[4, col] += 1
    col += 1
    
    # 2. bT B22 (1x2 * 2x2) -> C[2, 0:2] (bottom-left)
    # b0=A20 (U6), b1=A21 (U7)
    # B00 (V0), B01 (V1), B10 (V8), B11 (V9)
    # C20 (W6), C21 (W7)
    
    # b0 B00 -> C20
    U[6, col] += 1; V[0, col] += 1; W[6, col] += 1
    col += 1
    # b0 B01 -> C21
    U[6, col] += 1; V[1, col] += 1; W[7, col] += 1
    col += 1
    # b1 B10 -> C20
    U[7, col] += 1; V[8, col] += 1; W[6, col] += 1
    col += 1
    # b1 B11 -> C21
    U[7, col] += 1; V[9, col] += 1; W[7, col] += 1
    col += 1
    
    # 3. c eT (scalar * 1x2) -> C[2, 0:2]
    # c=A22 (U8)
    # e0=B20 (V16), e1=B21 (V17)
    # C20 (W6), C21 (W7)
    
    # c e0 -> C20
    U[8, col] += 1; V[16, col] += 1; W[6, col] += 1
    col += 1
    # c e1 -> C21
    U[8, col] += 1; V[17, col] += 1; W[7, col] += 1
    col += 1
    
    # 4. bT d (1x2 * 2x1) -> C[2,2]
    # b0=A20 (U6), b1=A21 (U7)
    # d0=B02 (V2), d1=B12 (V10)
    # C22 (W8)
    
    # b0 d0 -> C22
    U[6, col] += 1; V[2, col] += 1; W[8, col] += 1
    col += 1
    # b1 d1 -> C22
    U[7, col] += 1; V[10, col] += 1; W[8, col] += 1
    col += 1
    
    # 5. c f (scalar * scalar) -> C[2,2]
    U[8, col] += 1; V[18, col] += 1; W[8, col] += 1
    col += 1
    
    # 6. A22 d (2x2 * 2x1) -> C[0:2, 2]
    # A00 (U0), A01 (U1), A10 (U3), A11 (U4)
    # d0=B02 (V2), d1=B12 (V10)
    # C02 (W2), C12 (W5)
    
    # A00 d0 -> C02
    U[0, col] += 1; V[2, col] += 1; W[2, col] += 1
    col += 1
    # A01 d1 -> C02
    U[1, col] += 1; V[10, col] += 1; W[2, col] += 1
    col += 1
    # A10 d0 -> C12
    U[3, col] += 1; V[2, col] += 1; W[5, col] += 1
    col += 1
    # A11 d1 -> C12
    U[4, col] += 1; V[10, col] += 1; W[5, col] += 1
    col += 1
    
    # 7. a f (2x1 * scalar) -> C[0:2, 2]
    # a0=A02 (U2), a1=A12 (U5)
    # f=B22 (V18)
    # C02 (W2), C12 (W5)
    
    # a0 f -> C02
    U[2, col] += 1; V[18, col] += 1; W[2, col] += 1
    col += 1
    # a1 f -> C12
    U[5, col] += 1; V[18, col] += 1; W[5, col] += 1
    col += 1
    
    # Part 2: M_3,3,5 (Rank 45)
    # B columns 3-7
    for k in range(3, 8):
        for i in range(3):
            for j in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
