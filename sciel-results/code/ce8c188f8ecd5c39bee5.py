import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    # Strassen 2x2x2 decomposition (Rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices for 2x2: i,j in {0,1}
    # Flattened: i*2+j
    
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U2 (A coeffs), V2 (B coeffs), W2 (C coeffs)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1 # A11, A22
    V2[0, 0] = 1; V2[3, 0] = 1 # B11, B22
    W2[0, 0] = 1; W2[3, 0] = 1 # C11, C22
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1 # A21, A22
    V2[0, 1] = 1 # B11
    W2[2, 1] = 1; W2[3, 1] = -1 # C21, C22
    
    # M3
    U2[0, 2] = 1 # A11
    V2[1, 2] = 1; V2[3, 2] = -1 # B12, B22
    W2[1, 2] = 1; W2[3, 2] = 1 # C12, C22
    
    # M4
    U2[3, 3] = 1 # A22
    V2[2, 3] = 1; V2[0, 3] = -1 # B21, B11
    W2[0, 3] = 1; W2[2, 3] = 1 # C11, C21
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1 # A11, A12
    V2[3, 4] = 1 # B22
    W2[0, 4] = -1; W2[1, 4] = 1 # C11, C12
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1 # A21, A11
    V2[0, 5] = 1; V2[1, 5] = 1 # B11, B12
    W2[3, 5] = 1 # C22
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1 # A12, A22
    V2[2, 6] = 1; V2[3, 6] = 1 # B21, B22
    W2[0, 6] = 1 # C11
    
    # Kronecker product for 4x4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Part 2: Another 4x4x4
    # We need to map the k indices.
    # The full problem is 4x4x9.
    # k in 0..8.
    # Split k into k1 (0..3), k2 (4..7), k3 (8).
    # The tensor is sum of 3 parts.
    
    # We need to construct U, V, W for the full problem.
    # U is 16 x Rank
    # V is 36 x Rank
    # W is 36 x Rank
    
    rank1 = U4.shape[1] # 49
    rank2 = U4.shape[1] # 49
    rank3 = 16 # Schoolbook for 4x4x1
    
    total_rank = rank1 + rank2 + rank3
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Fill Part 1 (k in 0..3)
    # V indices: j*4 + k. j in 0..3, k in 0..3.
    # V4 has indices for 2x2. We need to map to 4x4.
    # V4 is 16x49. Indices are j*4+k?
    # In the Kronecker product, the indices are flattened.
    # U2 indices: i*2+j.
    # U4 indices: (i*2+j1)*4 + (i*2+j2)? No.
    # Kron(U2, U2) has shape (4*4, 7*7) = (16, 49).
    # The row index is i1*4 + i2?
    # Let's check np.kron indexing.
    # If A is m1 x n1, B is m2 x n2.
    # C = kron(A, B) is m1*m2 x n1*n2.
    # C[i1*m2 + i2, j1*n2 + j2] = A[i1, j1] * B[i2, j2].
    # So row index is i1*4 + i2.
    # For 4x4 matrix, index is i*4 + j.
    # So i1 corresponds to high bit, i2 to low bit.
    # This matches the 4x4 indexing.
    
    # So U4 rows correspond to 4x4 A indices.
    # V4 rows correspond to 4x4 B indices.
    # W4 rows correspond to 4x4 C indices.
    
    # For Part 1, k is 0..3.
    # V indices for Part 1: j*4 + k.
    # V4 row index is j*4 + k?
    # V4 is 16x49. Row index is j*4 + k.
    # So we can just copy V4 to V[0:16, 0:49].
    
    U[0:16, 0:rank1] = U4
    V[0:16, 0:rank1] = V4
    # W indices for Part 1: k*4 + i.
    # k in 0..3, i in 0..3.
    # W4 row index is k*4 + i.
    W[0:16, 0:rank1] = W4
    
    # Part 2 (k in 4..7)
    # V indices: j*4 + k. k = 4..7.
    # So V rows 16:32.
    # W indices: k*4 + i. k = 4..7.
    # So W rows 16:32.
    
    U[0:16, rank1:rank1+rank2] = U4
    V[16:32, rank1:rank1+rank2] = V4
    W[16:32, rank1:rank1+rank2] = W4
    
    # Part 3 (k in 8)
    # Schoolbook 4x4x1.
    # V indices: j*4 + 8. j in 0..3.
    # So V rows 32:36.
    # W indices: 8*4 + i. i in 0..3.
    # So W rows 32:36.
    
    for i in range(4):
        for j in range(4):
            c = i * 4 + j
            col = rank1 + rank2 + c
            U[i * 4 + j, col] = 1
            V[j * 4 + 8, col] = 1
            W[8 * 4 + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
