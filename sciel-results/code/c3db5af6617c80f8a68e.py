import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # We construct the rank-28 decomposition directly.
    # This is based on the block decomposition:
    # A (2x3) split into A1 (2x2) and A2 (2x1)
    # B (3x5) split into B1 (2x5) and B2 (1x5)
    # C = A1 B1 + A2 B2
    
    # A1 B1 (2x2 * 2x5) decomposes into:
    #   A1 B1[:,0:2] (2x2 * 2x2) -> Rank 7 (Strassen)
    #   A1 B1[:,2:4] (2x2 * 2x2) -> Rank 7 (Strassen)
    #   A1 B1[:,4:5] (2x2 * 2x1) -> Rank 4 (Schoolbook)
    # Total for A1 B1: 18
    
    # A2 B2 (2x1 * 1x5) decomposes into:
    #   A2 B2[:,0:2] (2x1 * 1x2) -> Rank 4 (Schoolbook)
    #   A2 B2[:,2:4] (2x1 * 1x2) -> Rank 4 (Schoolbook)
    #   A2 B2[:,4:5] (2x1 * 1x1) -> Rank 2 (Schoolbook)
    # Total for A2 B2: 10
    
    # Total Rank: 28

    rank = 28
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = (A10 + A11) B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = A00 (B01 - B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M4 = A11 (B10 - B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A00 + A01) B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A10 - A00)(B00 + B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A01 - A11)(B10 + B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    # Block 1: A[:,0:2] * B[0:2,0:2] (Strassen)
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[:,0:2] * B[0:2,2:4] (Strassen)
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[:,0:2] * B[0:2,4:5] (Schoolbook)
    add_schoolbook([0, 1], [0, 1], [4])
    
    # Block 4: A[:,2:3] * B[2:3,0:2] (Schoolbook)
    add_schoolbook([0, 1], [2], [0, 1])
    
    # Block 5: A[:,2:3] * B[2:3,2:4] (Schoolbook)
    add_schoolbook([0, 1], [2], [2, 3])
    
    # Block 6: A[:,2:3] * B[2:3,4:5] (Schoolbook)
    add_schoolbook([0, 1], [2], [4])

    assert col == rank, f"Rank mismatch: {col} != {rank}"

    # Verification
    T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    if not np.allclose(T_rec, T_target, atol=1e-9):
        # Fallback to schoolbook rank 30 if something went wrong
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
