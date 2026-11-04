import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Strassen decomposition for 2x2x2
    # Rank 7
    # U1: 4x7, V1: 4x7, W1: 4x7
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))
    
    # M1 = (A11 - A01) * (B00 + B10)
    # A11 is idx 3, A01 is idx 1
    # B00 is idx 0, B10 is idx 2
    U1[3, 0] = 1; U1[1, 0] = -1
    V1[0, 0] = 1; V1[2, 0] = 1
    # C00 += M1, C11 += M1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # M2 = (A00 + A11) * B10
    U1[0, 1] = 1; U1[3, 1] = 1
    V1[2, 1] = 1
    # C10 += M2, C11 -= M2
    W1[2, 1] = 1; W1[3, 1] = -1
    
    # M3 = A00 * (B01 - B11)
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    # C01 += M3, C11 += M3
    W1[1, 2] = 1; W1[3, 2] = 1
    
    # M4 = A11 * (B11 - B00)
    U1[3, 3] = 1
    V1[3, 3] = 1; V1[0, 3] = -1
    # C00 += M4, C10 += M4
    W1[0, 3] = 1; W1[2, 3] = 1
    
    # M5 = (A00 + A01) * B00
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[0, 4] = 1
    # C00 -= M5, C01 += M5
    W1[0, 4] = -1; W1[1, 4] = 1
    
    # M6 = (A10 - A00) * (B01 + B11)
    U1[2, 5] = 1; U1[0, 5] = -1
    V1[1, 5] = 1; V1[3, 5] = 1
    # C11 += M6
    W1[3, 5] = 1
    
    # M7 = (A01 - A10) * (B10 - B11)
    U1[1, 6] = 1; U1[2, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = -1
    # C00 += M7
    W1[0, 6] = 1
    
    # 2. Schoolbook decomposition for 2x2x4
    # Rank 8
    # U2: 4x8, V2: 8x8, W2: 8x4
    U2 = np.zeros((4, 8))
    V2 = np.zeros((8, 8))
    W2 = np.zeros((8, 8))
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U2[i*2 + j, col] = 1
                V2[j*4 + k, col] = 1
                W2[k*2 + i, col] = 1
                col += 1
                
    # 3. Kronecker Product for 4x4x8
    # U_full = U1 x U2 (4x4 x 56)
    # V_full = V1 x V2 (4x8 x 56)
    # W_full = W1 x W2 (8x4 x 56)
    
    U_full = np.kron(U1, U2)
    V_full = np.kron(V1, V2)
    W_full = np.kron(W1, W2)
    
    # 4. Extract 3x4x8
    # U: Take first 12 rows (i=0,1,2 for 4x4 matrix A)
    # Rows of U_full correspond to (i1, i2) where i = i1*2 + i2?
    # np.kron(U1, U2) row index is r1*4 + r2.
    # r1 is index in U1 (0..3), r2 is index in U2 (0..3).
    # U1 index r1 = i1*2 + j1.
    # U2 index r2 = i2*2 + j2.
    # The full A index is i = i1*2 + i2? No.
    # In Kronecker product of multiplications, the block structure is:
    # A_full[i1, i2, j1, j2] = A1[i1, j1] * A2[i2, j2]?
    # Actually, the standard embedding is A = A1 \otimes A2.
    # Row index of A is (i1, i2). Col index is (j1, j2).
    # Flat row index: i1 * N2 + i2.
    # Here N2=2. So i = i1 * 2 + i2.
    # We want i = 0, 1, 2.
    # i=0 -> i1=0, i2=0. Row in U_full: r1=0, r2=0?
    # Wait, U1 has rows for i1, j1. U2 has rows for i2, j2.
    # U_full row index is (i1, j1, i2, j2) flattened?
    # np.kron(U1, U2) creates a matrix where block (a,b) of U1 is U1[a,b]*U2.
    # So row index is a * rows(U2) + c.
    # a is row index of U1. c is row index of U2.
    # U1 row a = i1 * 2 + j1.
    # U2 row c = i2 * 2 + j2.
    # So U_full row index = (i1 * 2 + j1) * 4 + (i2 * 2 + j2).
    # This does NOT directly map to i = i1 * 2 + i2.
    # The Kronecker product of matrices A1 (n1 x m1) and A2 (n2 x m2) results in A (n1*n2 x m1*m2).
    # The row index of A is i = i1 * n2 + i2.
    # The column index of A is j = j1 * m2 + j2.
    # The tensor decomposition U, V, W must respect this indexing.
    # U_full row index should be i * m + j.
    # i = i1 * 2 + i2.
    # j = j1 * 2 + j2.
    # So row index = (i1 * 2 + i2) * 4 + (j1 * 2 + j2).
    # But np.kron(U1, U2) gives row index = (i1 * 2 + j1) * 4 + (i2 * 2 + j2).
    # This is a permutation of the rows!
    # We need to permute the rows of U_full to match the standard indexing.
    
    # Let's create the permutation.
    # Current row index r = (i1 * 2 + j1) * 4 + (i2 * 2 + j2).
    # Target row index t = (i1 * 2 + i2) * 4 + (j1 * 2 + j2).
    
    perm_U = []
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    t = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                    r = (i1 * 2 + j1) * 4 + (i2 * 2 + j2)
                    perm_U.append((t, r))
                    
    # Sort by target index t
    perm_U.sort(key=lambda x: x[0])
    row_indices_U = [r for t, r in perm_U]
    
    U_4x4x8 = U_full[row_indices_U, :]
    
    # Similarly for V.
    # V_full row index from kron: (j1 * 2 + k1) * 8 + (j2 * 2 + k2)?
    # V1 is 4x7 (m1=2, p1=2). Row index = j1 * 2 + k1.
    # V2 is 8x8 (m2=2, p2=4). Row index = j2 * 4 + k2.
    # V_full row index = (j1 * 2 + k1) * 8 + (j2 * 4 + k2).
    # Target V row index for 4x8 matrix B:
    # j = j1 * 2 + j2.
    # k = k1 * 4 + k2.
    # t = j * 8 + k = (j1 * 2 + j2) * 8 + (k1 * 4 + k2).
    
    perm_V = []
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(4):
                    t = (j1 * 2 + j2) * 8 + (k1 * 4 + k2)
                    r = (j1 * 2 + k1) * 8 + (j2 * 4 + k2)
                    perm_V.append((t, r))
    perm_V.sort(key=lambda x: x[0])
    row_indices_V = [r for t, r in perm_V]
    
    V_4x4x8 = V_full[row_indices_V, :]
    
    # Similarly for W.
    # W1 is 4x7 (p1=2, n1=2). Row index = k1 * 2 + i1.
    # W2 is 8x8 (p2=4, n2=2). Row index = k2 * 2 + i2.
    # W_full row index = (k1 * 2 + i1) * 8 + (k2 * 2 + i2).
    # Target W row index for 8x4 matrix C:
    # k = k1 * 4 + k2.
    # i = i1 * 2 + i2.
    # t = k * 4 + i = (k1 * 4 + k2) * 4 + (i1 * 2 + i2).
    
    perm_W = []
    for k1 in range(2):
        for k2 in range(4):
            for i1 in range(2):
                for i2 in range(2):
                    t = (k1 * 4 + k2) * 4 + (i1 * 2 + i2)
                    r = (k1 * 2 + i1) * 8 + (k2 * 2 + i2)
                    perm_W.append((t, r))
    perm_W.sort(key=lambda x: x[0])
    row_indices_W = [r for t, r in perm_W]
    
    W_4x4x8 = W_full[row_indices_W, :]
    
    # Now extract 3x4x8 from 4x4x8
    # U: 12 rows (i=0,1,2)
    U_3x4x8 = U_4x4x8[:12, :]
    
    # V: 32 rows (all)
    V_3x4x8 = V_4x4x8
    
    # W: 24 rows (i=0,1,2)
    # W_4x4x8 has rows indexed by k*4 + i.
    # We need W_3x4x8 rows indexed by k*3 + i.
    W_3x4x8 = np.zeros((24, 56))
    for k in range(8):
        for i in range(3):
            W_3x4x8[k*3 + i, :] = W_4x4x8[k*4 + i, :]
            
    return {"U": U_3x4x8.tolist(), "V": V_3x4x8.tolist(), "W": W_3x4x8.tolist()}
# EVOLVE-BLOCK-END
