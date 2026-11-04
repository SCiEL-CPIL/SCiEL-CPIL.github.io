import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # 1. Strassen for 4x4x4
    # U2, V2, W2 for 2x2x2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    # M6
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Kronecker product for 4x4x4
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    rank4 = U4.shape[1] # 49

    # 2. Schoolbook for 4x4x3
    n3, m3, p3 = 4, 4, 3
    rank3 = n3 * m3 * p3 # 48
    U3 = np.zeros((n3 * m3, rank3))
    V3 = np.zeros((m3 * p3, rank3))
    W3 = np.zeros((p3 * n3, rank3))
    
    col = 0
    for i in range(n3):
        for j in range(m3):
            for k in range(p3):
                U3[i * m3 + j, col] = 1
                V3[j * p3 + k, col] = 1
                W3[k * n3 + i, col] = 1
                col += 1

    # 3. Assemble
    total_rank = rank4 + rank3
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 4x4x4 (cols 0-3)
    U[:, :rank4] = U4
    # Map V4 rows (j*4+k, k in 0..3) to V rows (j*7+k)
    for j in range(4):
        for k in range(4):
            V[j * 7 + k, :rank4] = V4[j * 4 + k, :]
    # Map W4 rows (k*4+i, k in 0..3) to W rows (k*4+i)
    W[:, :rank4] = W4
    
    # Part 2: 4x4x3 (cols 4..6)
    U[:, rank4:] = U3
    # Map V3 rows (j*3+k, k in 0..2) to V rows (j*7+(k+4))
    for j in range(4):
        for k in range(3):
            V[j * 7 + (k + 4), rank4:] = V3[j * 3 + k, :]
    # Map W3 rows (k*4+i, k in 0..2) to W rows ((k+4)*4+i)
    for k in range(3):
        for i in range(4):
            W[(k + 4) * 4 + i, rank4:] = W3[k * 4 + i, :]

    return {"U": U, "V": V, "W": W}
