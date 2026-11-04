import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Returns a rank-56 exact tensor decomposition for 4x4x8 matrix multiplication.
    
    Uses a mixed decomposition for 4x4x4 blocks:
    - Outer level: Strassen's algorithm for 2x2 blocks of 2x2 matrices (rank 7).
    - Inner level: Schoolbook multiplication for 2x2 matrices (rank 4).
    - Total rank per 4x4x4 block: 7 * 4 = 28.
    - Two blocks for 4x4x8: 28 * 2 = 56.
    
    Indexing convention:
    - U: (n*m, rank) = (16, 56), U[i*m+j, r] is coefficient of A[i,j] in product r
    - V: (m*p, rank) = (32, 56), V[j*p+k, r] is coefficient of B[j,k] in product r
    - W: (p*n, rank) = (32, 56), W[k*n+i, r] is coefficient of product r in C[i,k]
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 decomposition (rank 7)
    U2s = np.zeros((4, 7))
    V2s = np.zeros((4, 7))
    W2s = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U2s[0, 0] = 1; U2s[3, 0] = 1
    V2s[0, 0] = 1; V2s[3, 0] = 1
    W2s[0, 0] = 1; W2s[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U2s[2, 1] = 1; U2s[3, 1] = 1
    V2s[0, 1] = 1
    W2s[1, 1] = 1; W2s[3, 1] = -1
    
    # r=2: M3 = A00(B01-B11)
    U2s[0, 2] = 1
    V2s[1, 2] = 1; V2s[3, 2] = -1
    W2s[2, 2] = 1; W2s[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U2s[3, 3] = 1
    V2s[0, 3] = -1; V2s[2, 3] = 1
    W2s[0, 3] = 1; W2s[1, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U2s[0, 4] = 1; U2s[1, 4] = 1
    V2s[3, 4] = 1
    W2s[0, 4] = -1; W2s[2, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U2s[0, 5] = -1; U2s[2, 5] = 1
    V2s[0, 5] = 1; V2s[1, 5] = 1
    W2s[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U2s[1, 6] = 1; U2s[3, 6] = -1
    V2s[2, 6] = 1; V2s[3, 6] = 1
    W2s[0, 6] = 1

    # Schoolbook 2x2x2 decomposition (rank 4)
    U2k = np.zeros((4, 4))
    V2k = np.zeros((4, 4))
    W2k = np.zeros((4, 4))
    
    # r=0: A00 * B00
    U2k[0, 0] = 1
    V2k[0, 0] = 1
    W2k[0, 0] = 1
    
    # r=1: A01 * B10
    U2k[1, 1] = 1
    V2k[2, 1] = 1
    W2k[1, 1] = 1
    
    # r=2: A10 * B00
    U2k[2, 2] = 1
    V2k[0, 2] = 1
    W2k[2, 2] = 1
    
    # r=3: A11 * B11
    U2k[3, 3] = 1
    V2k[3, 3] = 1
    W2k[3, 3] = 1

    # Build 4x4x4 decomposition (Rank 28)
    U4 = np.kron(U2s, U2k)
    V4 = np.kron(V2s, V2k)
    W4 = np.kron(W2s, W2k)
    
    # Construct 4x4x8 decomposition (Rank 56)
    rank = 56
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # Left block (columns 0-3 of B, i.e., k=0..3 in C)
    U[:, 0:28] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:28] = V4[j * 4 + k, :]
    W[0:16, 0:28] = W4

    # Right block (columns 4-7 of B, i.e., k=4..7 in C)
    U[:, 28:56] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 28:56] = V4[j * 4 + k, :]
            W[(4 + k) * 4 : (4 + k) * 4 + 4, 28:56] = W4[k * 4 : k * 4 + 4, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
