"""Rank-97 exact tensor decomposition for 4x4 * 4x7 matrix multiplication.

Approach:
- Split the p=7 dimension into p1=4 and p2=3.
- For the 4x4x4 sub-problem, use a rank-49 Strassen-based decomposition
  (tensor product of two rank-7 Strassen decompositions for 2x2x2).
- For the 4x4x3 sub-problem, use schoolbook rank 48.
- Total rank: 49 + 48 = 97.

The W indexing follows the official trace convention: W[k*n + i, r].
"""

import numpy as np

DIMENSIONS = [4, 4, 7]


def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) where:
    - U[i*2 + j, r] corresponds to A[i,j]
    - V[j*2 + k, r] corresponds to B[j,k]
    - W[k*2 + i, r] corresponds to C[i,k]
    
    The 7 Strassen products are:
    M0 = (A00+A11)*(B00+B11)
    M1 = (A10+A11)*B00
    M2 = A00*(B01-B11)
    M3 = A11*(B10-B00)
    M4 = (A00+A01)*B11
    M5 = (A10-A00)*(B00+B01)
    M6 = (A01-A11)*(B10+B11)
    
    Then:
    C00 = M0 + M3 - M4 + M6
    C01 = M2 + M4
    C10 = M1 + M3
    C11 = M0 - M1 + M2 + M5
    """
    # U: 4x7, index [i*2+j, r]
    # V: 4x7, index [j*2+k, r]
    # W: 4x7, index [k*2+i, r]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M0 = (A00+A11)*(B00+B11)
    # U: A00=1, A11=1 -> indices 0 and 3
    U[0, 0] = 1
    U[3, 0] = 1
    # V: B00=1, B11=1 -> indices 0 and 3
    V[0, 0] = 1
    V[3, 0] = 1
    # W: contributes to C00 (k=0,i=0 -> idx 0) and C11 (k=1,i=1 -> idx 3)
    W[0, 0] = 1   # C00
    W[3, 0] = 1   # C11
    
    # M1 = (A10+A11)*B00
    # U: A10=1, A11=1 -> indices 2 and 3
    U[2, 1] = 1
    U[3, 1] = 1
    # V: B00=1 -> index 0
    V[0, 1] = 1
    # W: contributes to C10 (k=0,i=1 -> idx 2) and C11 (k=1,i=1 -> idx 3) with -1
    W[2, 1] = 1   # C10
    W[3, 1] = -1  # C11
    
    # M2 = A00*(B01-B11)
    # U: A00=1 -> index 0
    U[0, 2] = 1
    # V: B01=1, B11=-1 -> indices 1 and 3
    V[1, 2] = 1
    V[3, 2] = -1
    # W: contributes to C00 (k=0,i=0 -> idx 0) and C01 (k=1,i=0 -> idx 1)
    W[0, 2] = 1   # C00
    W[1, 2] = 1   # C01
    
    # M3 = A11*(B10-B00)
    # U: A11=1 -> index 3
    U[3, 3] = 1
    # V: B10=1, B00=-1 -> indices 2 and 0
    V[2, 3] = 1
    V[0, 3] = -1
    # W: contributes to C00 (k=0,i=0 -> idx 0) and C10 (k=0,i=1 -> idx 2)
    W[0, 3] = 1   # C00
    W[2, 3] = 1   # C10
    
    # M4 = (A00+A01)*B11
    # U: A00=1, A01=1 -> indices 0 and 1
    U[0, 4] = 1
    U[1, 4] = 1
    # V: B11=1 -> index 3
    V[3, 4] = 1
    # W: contributes to C00 (k=0,i=0 -> idx 0) with -1 and C01 (k=1,i=0 -> idx 1)
    W[0, 4] = -1  # C00
    W[1, 4] = 1   # C01
    
    # M5 = (A10-A00)*(B00+B01)
    # U: A10=1, A00=-1 -> indices 2 and 0
    U[2, 5] = 1
    U[0, 5] = -1
    # V: B00=1, B01=1 -> indices 0 and 1
    V[0, 5] = 1
    V[1, 5] = 1
    # W: contributes to C11 (k=1,i=1 -> idx 3)
    W[3, 5] = 1   # C11
    
    # M6 = (A01-A11)*(B10+B11)
    # U: A01=1, A11=-1 -> indices 1 and 3
    U[1, 6] = 1
    U[3, 6] = -1
    # V: B10=1, B11=1 -> indices 2 and 3
    V[2, 6] = 1
    V[3, 6] = 1
    # W: contributes to C00 (k=0,i=0 -> idx 0)
    W[0, 6] = 1   # C00
    
    return U, V, W


def _construct_4x4x4_strassen():
    """
    Construct rank-49 decomposition for 4x4 * 4x4 using tensor product of two
    rank-7 Strassen decompositions.
    
    Returns U, V, W each of shape (16, 49).
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    for r1 in range(7):
        for r2 in range(7):
            R = r1 * 7 + r2
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            u_val = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                            if u_val != 0:
                                U[i * 4 + j, R] += u_val
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            v_val = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                            if v_val != 0:
                                V[j * 4 + k, R] += v_val
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            w_val = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
                            if w_val != 0:
                                W[k * 4 + i, R] += w_val
    
    return U, V, W


def construct():
    """
    Construct a rank-97 exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Splits p=7 into p1=4 (Strassen rank 49) and p2=3 (schoolbook rank 48).
    Total rank: 97.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B)
    U1, V1, W1 = _construct_4x4x4_strassen()
    
    # Part 2: 4x4 * 4x3 (columns 4-6 of B), schoolbook
    rank2 = 4 * 4 * 3  # 48
    U2 = np.zeros((16, rank2), dtype=int)
    V2 = np.zeros((16, rank2), dtype=int)
    W2 = np.zeros((16, rank2), dtype=int)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U2[i * 4 + j, col] = 1
                V2[j * 7 + k, col] = 1
                W2[k * 4 + i, col] = 1
                col += 1
    
    # Combine
    rank = 49 + rank2
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    U[:, :49] = U1
    V[:, :49] = V1
    W[:, :49] = W1
    
    U[:, 49:] = U2
    V[:, 49:] = V2
    W[:, 49:] = W2
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
