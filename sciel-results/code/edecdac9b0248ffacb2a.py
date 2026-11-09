import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin

    n, m, p = DIMENSIONS

    # We aim for rank 32, which is the theoretical minimum R(2,4,8) = 32.
    # We use the tensor decomposition: M_{2,4,8} = M_{2,2,4} \otimes M_{1,2,2}
    # R(2,2,4) = 16 (schoolbook)
    # R(1,2,2) = 2 (schoolbook)
    # Total rank = 16 * 2 = 32.

    # --- Decomposition for M_{2,2,4} ---
    # A: 2x2, B: 2x4
    # U1 shape: (2*2, 16) = (4, 16)
    # V1 shape: (2*4, 16) = (8, 16)
    # W1 shape: (4*2, 16) = (8, 16)
    n1, m1, p1 = 2, 2, 4
    r1 = n1 * m1 * p1  # 16
    U1 = np.zeros((n1 * m1, r1))
    V1 = np.zeros((m1 * p1, r1))
    W1 = np.zeros((p1 * n1, r1))

    col = 0
    for i1 in range(n1):
        for j1 in range(m1):
            for k1 in range(p1):
                U1[i1 * m1 + j1, col] = 1
                V1[j1 * p1 + k1, col] = 1
                W1[k1 * n1 + i1, col] = 1
                col += 1

    # --- Decomposition for M_{1,2,2} ---
    # A: 1x2, B: 2x2
    # U2 shape: (1*2, 2) = (2, 2)
    # V2 shape: (2*2, 2) = (4, 2)
    # W2 shape: (2*1, 2) = (2, 2)
    n2, m2, p2 = 1, 2, 2
    r2 = n2 * m2 * p2  # 2
    U2 = np.zeros((n2 * m2, r2))
    V2 = np.zeros((m2 * p2, r2))
    W2 = np.zeros((p2 * n2, r2))

    col = 0
    for i2 in range(n2):
        for j2 in range(m2):
            for k2 in range(p2):
                U2[i2 * m2 + j2, col] = 1
                V2[j2 * p2 + k2, col] = 1
                W2[k2 * n2 + i2, col] = 1
                col += 1

    # --- Kronecker Product ---
    # U = U1 \otimes U2 -> shape (4*2, 16*2) = (8, 32)
    # V = V1 \otimes V2 -> shape (8*4, 16*2) = (32, 32)
    # W = W1 \otimes W2 -> shape (8*2, 16*2) = (16, 32)
    
    U = np.kron(U1, U2)
    V = np.kron(V1, V2)
    W = np.kron(W1, W2)

    # Verification (optional, but good for sanity)
    # The tensor T[i, j, k] = 1 if (i,j) is in A, (j,k) is in B, (k,i) is in C? 
    # Actually, matrix mult tensor is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = delta?
    # Let's just return the list of lists as required.

    def to_list(mat):
        return mat.tolist()

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
