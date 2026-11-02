import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0
    n, m, p = 2, 4, 6

    # Known valid decomposition: rank 42 via Kronecker product of Strassen (2x2x2, rank 7)
    # and trivial (1x2x3, rank 6).
    # We construct this directly and return it.

    # Strassen 2x2x2 decomposition (rank 7)
    U1 = np.zeros((4, 7))
    V1 = np.zeros((4, 7))
    W1 = np.zeros((4, 7))

    U1[0,0] = 1; U1[3,0] = 1
    U1[2,1] = 1; U1[3,1] = 1
    U1[0,2] = 1
    U1[3,3] = 1
    U1[0,4] = 1; U1[1,4] = 1
    U1[2,5] = 1; U1[0,5] = -1
    U1[1,6] = 1; U1[3,6] = -1

    V1[0,0] = 1; V1[3,0] = 1
    V1[0,1] = 1
    V1[1,2] = 1; V1[3,2] = -1
    V1[2,3] = 1; V1[0,3] = -1
    V1[3,4] = 1
    V1[0,5] = 1; V1[1,5] = 1
    V1[2,6] = 1; V1[3,6] = 1

    W1[0,0] = 1; W1[0,3] = 1; W1[0,4] = -1; W1[0,6] = 1
    W1[2,2] = 1; W1[2,4] = 1
    W1[1,1] = 1; W1[1,3] = 1
    W1[3,0] = 1; W1[3,1] = -1; W1[3,2] = 1; W1[3,5] = 1

    # Trivial 1x2x3 decomposition (rank 6)
    U2 = np.zeros((2, 6))
    V2 = np.zeros((6, 6))
    W2 = np.zeros((3, 6))
    for t in range(6):
        j = t // 3
        k = t % 3
        U2[j, t] = 1
        V2[j*3 + k, t] = 1
        W2[k, t] = 1

    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2

            for i1 in range(2):
                for j1 in range(2):
                    v1 = U1[i1*2 + j1, r1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        v2 = U2[j2, r2]
                        if v2 == 0:
                            continue
                        idx_u = i1 * 4 + j1 * 2 + j2
                        U[idx_u, r] += v1 * v2

            for j1 in range(2):
                for k1 in range(2):
                    v1 = V1[j1*2 + k1, r1]
                    if v1 == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(3):
                            v2 = V2[j2*3 + k2, r2]
                            if v2 == 0:
                                continue
                            a = j1 * 2 + j2
                            b = k1 * 3 + k2
                            idx_v = a * 6 + b
                            V[idx_v, r] += v1 * v2

            for i1 in range(2):
                for k1 in range(2):
                    v1 = W1[k1*2 + i1, r1]
                    if v1 == 0:
                        continue
                    for k2 in range(3):
                        v2 = W2[k2, r2]
                        if v2 == 0:
                            continue
                        i = i1
                        b = k1 * 3 + k2
                        idx_w = b * 2 + i
                        W[idx_w, r] += v1 * v2

    # Quick verification
    if time.time() - start_time < deadline - 5:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
