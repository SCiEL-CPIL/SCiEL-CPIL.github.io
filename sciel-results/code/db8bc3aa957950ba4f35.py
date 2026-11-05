import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # We attempt to find a rank-28 decomposition using a block structure:
    # - Two 2x2x2 Strassen blocks (rank 7 each) for the main 2x2 sub-blocks
    # - Schoolbook terms for the remaining parts
    # Total: 7 + 7 + 4 + 4 + 4 + 2 = 28

    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1
        W[k2*n+i2, col] = -1
        col += 1

        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1
        V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1
        V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1
        W[k1*n+i2, col] = 1
        col += 1

        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m+j1, col] = 1
        U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1
        W[k2*n+i1, col] = 1
        col += 1

        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        U[i2*m+j1, col] = 1
        U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1
        V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1

        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        U[i1*m+j2, col] = 1
        U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1
        V[j2*p+k2, col] = 1
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

    # Block decomposition:
    # A is 2x3, B is 3x5, C is 2x5
    # Split A columns: [0,1] and [2]
    # Split B rows: [0,1] and [2]
    # Split B columns: [0,1], [2,3], [4]
    # Split C columns: [0,1], [2,3], [4]

    # A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2]  (2x2x2, Strassen rank 7)
    add_strassen(0, 1, 0, 1, 0, 1)

    # A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4]  (2x2x2, Strassen rank 7)
    add_strassen(0, 1, 0, 1, 2, 3)

    # A[:,0:2] * B[0:2,4:5] -> C[0:2,4:5]  (2x2x1, schoolbook rank 4)
    add_schoolbook([0, 1], [0, 1], [4])

    # A[:,2:3] * B[2:3,0:2] -> C[0:2,0:2]  (2x1x2, schoolbook rank 4)
    add_schoolbook([0, 1], [2], [0, 1])

    # A[:,2:3] * B[2:3,2:4] -> C[0:2,2:4]  (2x1x2, schoolbook rank 4)
    add_schoolbook([0, 1], [2], [2, 3])

    # A[:,2:3] * B[2:3,4:5] -> C[0:2,4:5]  (2x1x1, schoolbook rank 2)
    add_schoolbook([0, 1], [2], [4])

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify correctness
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c].reshape(n, m)
        vc = V[:, c].reshape(m, p)
        wc = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]

    if not np.allclose(tensor, 1.0, atol=1e-9):
        # Fallback: full schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
