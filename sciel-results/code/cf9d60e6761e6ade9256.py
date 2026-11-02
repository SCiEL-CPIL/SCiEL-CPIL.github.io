import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _verify_exact(U, V, W):
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    # entries must be integer or half-integer
    for arr in (U, V, W):
        if not np.allclose(arr * 2.0, np.round(arr * 2.0), atol=1e-9):
            return False
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] must equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
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

def _strassen_444():
    # 4x4x4 Strassen decomposition: T[i,j,k] = delta_{i,k}
    n = 4
    U = [[0] * 49 for _ in range(n * n)]
    V = [[0] * 49 for _ in range(n * n)]
    W = [[0] * 49 for _ in range(n * n)]
    for c in range(8):
        base = 7 * c
        U[16][base] = 1; U[17][base] = 1; U[20][base] = 1; U[21][base] = 1
        U[24][base] = 1; U[25][base] = 1; U[28][base] = 1; U[29][base] = 1
        V[0][base] = 1; V[1][base] = 1; V[4][base] = 1; V[5][base] = 1
        V[8][base] = 1; V[9][base] = 1; V[12][base] = 1; V[13][base] = 1
        W[0][base] = 1; W[1][base] = 1; W[4][base] = 1; W[5][base] = 1
        W[8][base] = 1; W[9][base] = 1; W[12][base] = 1; W[13][base] = 1
        if c == 1:
            U[20][base] = -1; U[21][base] = -1
            W[16][base] = 1; W[17][base] = 1; W[20][base] = 1; W[21][base] = 1
            W[24][base] = 1; W[25][base] = 1; W[28][base] = 1; W[29][base] = 1
        elif c == 2:
            U[20][base] = -1; U[21][base] = -1
            W[0][base] = -1; W[1][base] = -1; W[4][base] = -1; W[5][base] = -1
            W[8][base] = -1; W[9][base] = -1; W[12][base] = -1; W[13][base] = -1
        elif c == 3:
            U[24][base] = -1; U[25][base] = -1; U[28][base] = -1; U[29][base] = -1
            V[8][base] = -1; V[9][base] = -1; V[12][base] = -1; V[13][base] = -1
            W[16][base] = 1; W[17][base] = 1; W[20][base] = 1; W[21][base] = 1
            W[24][base] = 1; W[25][base] = 1; W[28][base] = 1; W[29][base] = 1
        elif c == 4:
            U[24][base] = -1; U[25][base] = -1; U[28][base] = -1; U[29][base] = -1
            V[8][base] = -1; V[9][base] = -1; V[12][base] = -1; V[13][base] = -1
            W[0][base] = -1; W[1][base] = -1; W[4][base] = -1; W[5][base] = -1
            W[8][base] = -1; W[9][base] = -1; W[12][base] = -1; W[13][base] = -1
        elif c == 5:
            U[16][base] = 1; U[17][base] = 1; U[20][base] = 1; U[21][base] = 1
            U[24][base] = -1; U[25][base] = -1; U[28][base] = -1; U[29][base] = -1
            V[0][base] = 1; V[1][base] = 1; V[4][base] = 1; V[5][base] = 1
            V[8][base] = -1; V[9][base] = -1; V[12][base] = -1; V[13][base] = -1
            W[0][base] = 1; W[1][base] = 1; W[4][base] = 1; W[5][base] = 1
            W[16][base] = 1; W[17][base] = 1; W[20][base] = 1; W[21][base] = 1
        elif c == 6:
            U[16][base] = 1; U[17][base] = 1; U[20][base] = 1; U[21][base] = 1
            U[24][base] = -1; U[25][base] = -1; U[28][base] = -1; U[29][base] = -1
            V[0][base] = 1; V[1][base] = 1; V[4][base] = 1; V[5][base] = 1
            V[8][base] = -1; V[9][base] = -1; V[12][base] = -1; V[13][base] = -1
            W[4][base] = 1; W[5][base] = 1; W[8][base] = 1; W[9][base] = 1
            W[20][base] = 1; W[21][base] = 1; W[24][base] = 1; W[25][base] = 1
        elif c == 7:
            U[16][base] = 1; U[17][base] = 1; U[20][base] = 1; U[21][base] = 1
            U[24][base] = -1; U[25][base] = -1; U[28][base] = -1; U[29][base] = -1
            V[0][base] = 1; V[1][base] = 1; V[4][base] = 1; V[5][base] = 1
            V[8][base] = -1; V[9][base] = -1; V[12][base] = -1; V[13][base] = -1
            W[0][base] = -1; W[1][base] = -1; W[4][base] = -1; W[5][base] = -1
            W[8][base] = -1; W[9][base] = -1; W[12][base] = -1; W[13][base] = -1
    return U, V, W

def _strassen_449():
    n, m, p = DIMENSIONS
    rank = 49 * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    sU, sV, sW = _strassen_444()
    for c in range(49):
        for k in range(p):
            col = c * p + k
            for i in range(n):
                U[i * m + c][col] = sU[i * m + c][c]
                W[k * n + i][col] = sW[i * m + c][c]
                V[c * p + k][col] = sV[c * m + c][c]
    return {"U": U, "V": V, "W": W}

def _random_int_search(rank, n, m, p, iters, deadline):
    # random integer decomposition, keep best if exact
    best = None
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = np.random.randint(-2, 3, size=(n * m, rank))
        V = np.random.randint(-2, 3, size=(m * p, rank))
        W = np.random.randint(-2, 3, size=(p * n, rank))
        if _verify_exact(U, V, W):
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            break
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS

    school = _schoolbook()
    best = school
    best_rank = n * m * p

    strassen = _strassen_449()
    if _verify_exact(strassen["U"], strassen["V"], strassen["W"]):
        best = strassen
        best_rank = 49 * p
    else:
        strassen = None

    # time-guarded search for a strictly lower-rank exact integer decomposition
    if time.time() < deadline - 5.0:
        lo = max(1, best_rank - 1)
        for rank in range(lo, best_rank):
            if time.time() > deadline - 5.0:
                break
            res = _random_int_search(rank, n, m, p, iters=200, deadline=deadline - 5.0)
            if res is not None:
                best = res
                best_rank = rank
                break

    return best
