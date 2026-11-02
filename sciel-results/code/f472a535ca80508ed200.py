import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Strategy:
    # Part 1: j=0 handled by schoolbook: 12 terms
    # Part 2: j=1..4 split into two 2-wide blocks, each with 3 k-pairs,
    #         each using Strassen (7 terms) => 2*3*7 = 42 terms
    # Total rank = 54

    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: j = 0, all i, k (12 terms)
    for i in range(n):
        for k in range(p):
            U[i * m][col] = 1
            V[k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: two blocks of j-pairs: (1,2) and (3,4)
    for block in range(2):
        jb = 1 + block * 2
        j1, j2 = jb, jb + 1
        for s in range(3):
            kb = s * 2
            k1, k2 = kb, kb + 1

            u00 = 0 * m + j1
            u01 = 0 * m + j2
            u10 = 1 * m + j1
            u11 = 1 * m + j2
            v00 = j1 * p + k1
            v01 = j1 * p + k2
            v10 = j2 * p + k1
            v11 = j2 * p + k2
            w00 = k1 * n + 0
            w01 = k2 * n + 0
            w10 = k1 * n + 1
            w11 = k2 * n + 1

            # P1 = (a00+a11)(b00+b11)
            U[u00][col] += 1; U[u11][col] += 1
            V[v00][col] += 1; V[v11][col] += 1
            W[w00][col] += 1; W[w11][col] += 1
            col += 1

            # P2 = (a10+a11)b00
            U[u10][col] += 1; U[u11][col] += 1
            V[v00][col] += 1
            W[w10][col] += 1; W[w11][col] += -1
            col += 1

            # P3 = a00(b01-b11)
            U[u00][col] += 1
            V[v01][col] += 1; V[v11][col] += -1
            W[w01][col] += 1; W[w11][col] += 1
            col += 1

            # P4 = a11(b10-b00)
            U[u11][col] += 1
            V[v10][col] += 1; V[v00][col] += -1
            W[w00][col] += 1; W[w10][col] += 1
            col += 1

            # P5 = (a00+a01)b11
            U[u00][col] += 1; U[u01][col] += 1
            V[v11][col] += 1
            W[w00][col] += -1; W[w01][col] += 1
            col += 1

            # P6 = (a10-a00)(b00+b01)
            U[u10][col] += 1; U[u00][col] += -1
            V[v00][col] += 1; V[v01][col] += 1
            W[w11][col] += 1
            col += 1

            # P7 = (a01-a11)(b10+b11)
            U[u01][col] += 1; U[u11][col] += -1
            V[v10][col] += 1; V[v11][col] += 1
            W[w00][col] += 1
            col += 1

    # Verify
    def _check(Um, Vm, Wm):
        Ua = np.array(Um, dtype=np.int64).reshape(n, m, -1)
        Va = np.array(Vm, dtype=np.int64).reshape(m, p, -1)
        Wa = np.array(Wm, dtype=np.int64).reshape(p, n, -1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if int(np.sum(Ua[i, j] * Va[j, k] * Wa[k, i])) != 1:
                        return False
        return True

    if not _check(U, V, W):
        rf = n * m * p
        Uf = [[0] * rf for _ in range(n * m)]
        Vf = [[0] * rf for _ in range(m * p)]
        Wf = [[0] * rf for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j][c] = 1
                    Vf[j * p + k][c] = 1
                    Wf[k * n + i][c] = 1
                    c += 1
        return {"U": Uf, "V": Vf, "W": Wf}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
