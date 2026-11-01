import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 10.0  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 13

    # Direct rank-52 decomposition: 4 terms per column
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # A index: i*m+j = 2*i+j
    # B index: j*p+k = 13*j+k
    # C index (W): k*n+i = 2*k+i

    rank = 4 * p  # 52
    U = [[0] * rank for _ in range(n * m)]  # 4 x 52
    V = [[0] * rank for _ in range(m * p)]  # 26 x 52
    W = [[0] * rank for _ in range(p * n)]  # 26 x 52

    r = 0
    for k in range(p):
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]

        # Term: A[0,0] * B[0,k] -> C[0,k]
        U[0][r] = 1
        V[0 * p + k][r] = 1
        W[k * n + 0][r] = 1
        r += 1

        # Term: A[0,1] * B[1,k] -> C[0,k]
        U[1][r] = 1
        V[1 * p + k][r] = 1
        W[k * n + 0][r] = 1
        r += 1

        # Term: A[1,0] * B[0,k] -> C[1,k]
        U[2][r] = 1
        V[0 * p + k][r] = 1
        W[k * n + 1][r] = 1
        r += 1

        # Term: A[1,1] * B[1,k] -> C[1,k]
        U[3][r] = 1
        V[1 * p + k][r] = 1
        W[k * n + 1][r] = 1
        r += 1

        if time.time() - start_time > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
