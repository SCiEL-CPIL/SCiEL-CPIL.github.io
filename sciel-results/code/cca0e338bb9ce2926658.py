import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe (120s wall clock)

    # Strassen decomposition for 2x2x2: rank 7
    # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Tensor T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1 for all i,j,k in {0,1}
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00+A11)(B00+B11)
    # U: A00, A11; V: B00, B11; W: C00, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10+A11)B00
    # U: A10, A11; V: B00; W: C01, -C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # M2 = A00(B10-B11)
    # U: A00; V: B10, -B11; W: C00
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[0, 2] = 1

    # M3 = A11(B01-B10)
    # U: A11; V: B01, -B10; W: C10, C11
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[1, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = 1

    # M4 = (A00+A01)B11
    # U: A00, A01; V: B11; W: -C00, C01
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10-A00)(B00+B01)
    # U: A10, -A00; V: B00, B01; W: C10, C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[2, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1

    # M6 = (A01-A11)(B10+B11)
    # U: A01, -A11; V: B10, B11; W: C00
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[1, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Verify 2x2 Strassen
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U2[i*2+j, c] * V2[j*2+k, c] * W2[k*2+i, c] for c in range(7))
                if s != 1:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break

    if not valid2:
        # Fallback to schoolbook rank-64
        n, m, p = 4, 4, 4
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Construct 4x4x4 via Kronecker product of the two 2x2 Strassen decompositions
    # M4 = M2 ⊗ M2, rank ≤ 7*7 = 49
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for i in range(n):
        i1 = i // 2
        i2 = i % 2
        for j in range(m):
            j1 = j // 2
            j2 = j % 2
            u_row = i * m + j
            U[u_row, :] = np.kron(U2[i1 * 2 + j1, :], U2[i2 * 2 + j2, :])

    for j in range(m):
        j1 = j // 2
        j2 = j % 2
        for k in range(p):
            k1 = k // 2
            k2 = k % 2
            v_row = j * p + k
            V[v_row, :] = np.kron(V2[j1 * 2 + k1, :], V2[j2 * 2 + k2, :])

    for k in range(p):
        k1 = k // 2
        k2 = k % 2
        for i in range(n):
            i1 = i // 2
            i2 = i % 2
            w_row = k * n + i
            W[w_row, :] = np.kron(W2[k1 * 2 + i1, :], W2[k2 * 2 + i2, :])

    # Verify the 4x4x4 decomposition
    valid4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = sum(U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c] for c in range(49))
                if s != 1:
                    valid4 = False
                    break
            if not valid4:
                break
        if not valid4:
            break

    if not valid4:
        # Fallback to schoolbook rank-64
        rank = 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
