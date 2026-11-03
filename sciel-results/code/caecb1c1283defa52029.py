import time

def construct():
    """
    Matrix multiplication 2x3 by 3x8.
    Rank-44 decomposition:
      - 16 scalar terms for the j=0 column of A times the j=0 row of B
      - 4 blocks of 2x2x2 Strassen-style (7 terms each) for j=1,2
    Total rank = 16 + 4*7 = 44
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 2, 3, 8

    rank = 16 + 4 * 7  # 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    col = 0

    # Part 1: j=0, scalar terms: A[i,0] * B[0,k] -> C[i,k]
    for k in range(p):
        if time.time() > deadline:
            break
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 16

    # Part 2: 4 blocks of 2x2x2 using 7-term Strassen decomposition
    # For each block b: rows i=0,1; columns j=1,2; output cols k0=2b, k1=2b+1
    for b in range(4):
        if time.time() > deadline:
            break
        k0 = 2 * b
        k1 = 2 * b + 1

        # P1: (A[0,1]+A[1,2]) * (B[1,k0]+B[2,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # P2: (A[1,1]+A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # P3: A[0,1] * (B[1,k1]-B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # P4: A[1,2] * (B[2,k0]-B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # P5: (A[0,1]+A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        # P6: (A[1,1]-A[0,1]) * (B[1,k0]+B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # P7: (A[0,2]-A[1,2]) * (B[2,k0]+B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
