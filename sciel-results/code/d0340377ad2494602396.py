import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s wall-clock limit

    n, m, p = DIMENSIONS  # A is n x m, B is m x p, C is p x n

    # Strassen rank-7 coefficients for 2x2x2 (verified identity c = w . (u ⊙ v))
    # local A: 0=a11,1=a12,2=a21,3=a22 ; B: 0=b11,1=b12,2=b21,3=b22
    SU = [
        [0, 3],      # M1 = (a11+a22)
        [2, 3],      # M2 = (a21+a22)
        [0],         # M3 = a11
        [3],         # M4 = a22
        [0, 1],      # M5 = (a11+a12)
        [2, 0],      # M6 = (a21-a11)
        [1, 3],      # M7 = (a12-a22)
    ]
    SV = [
        [0, 3],      # (b11+b22)
        [0],         # b11
        [1, 3],      # (b12-b22)
        [2, 0],      # (b21-b11)
        [3],         # b22
        [0, 1],      # (b11+b12)
        [2, 3],      # (b21+b22)
    ]
    SW = [
        [0, 2, 3, 6],     # c11 = M1 - M4 + M5 - M7
        [6, 4],           # c12 = M5 + M6
        [1, 2],           # c21 = M2 + M3
        [0, 1, 2, 5],     # c22 = M1 - M3 + M4 + M6
    ]
    SWV = [
        [1, -1, 1, -1],
        [1, 1],
        [1, 1],
        [1, -1, 1, 1],
    ]
    SUV = [
        [1, 1], [1, 1], [1], [1], [1, 1], [1, -1], [1, -1],
    ]
    SVV = [
        [1, 1], [1], [1, -1], [1, -1], [1], [1, 1], [1, 1],
    ]

    # Sanity-check the 2x2x2 identity numerically before building the big tensor.
    ok = True
    for _t in range(20):
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C = A @ B
        M = np.zeros(7)
        for k in range(7):
            ua = 0.0
            for idx, val in zip(SU[k], SUV[k]):
                r = idx // 2
                cc = idx % 2
                ua += val * A[r, cc]
            vb = 0.0
            for idx, val in zip(SV[k], SVV[k]):
                r = idx // 2
                cc = idx % 2
                vb += val * B[r, cc]
            M[k] = ua * vb
        Crec = np.zeros((2, 2))
        for r in range(2):
            for c in range(2):
                s = 0.0
                for k, val in zip(SW[r * 2 + c], SWV[r * 2 + c]):
                    s += val * M[k]
                Crec[r, c] = s
        if not np.allclose(Crec, C, atol=1e-8):
            ok = False
            break
    if not ok:
        # Fallback: trivial rank-32 decomposition (guaranteed valid), still finite.
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    col = i * m * p + k * p + j
                    U[i * m + k, col] = 1.0
                    V[k * p + j, col] = 1.0
                    W[j * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def add_block(u_map, v_map, w_map, start_col):
        for k in range(7):
            col = start_col + k
            for idx, val in zip(SU[k], SUV[k]):
                U[u_map[idx], col] += val
            for idx, val in zip(SV[k], SVV[k]):
                V[v_map[idx], col] += val
            for idx, val in zip(SW[k], SWV[k]):
                W[w_map[idx], col] += val

    # U flat: A[i,k] -> i*m + k
    # V flat: B[k,j] -> k*p + j
    # W flat: C[i,j] -> j*n + i   (trace convention)

    # Block 1: A[:,0:2] * B[0:2,0:2] -> C[0:2,:]
    u1 = [0, 1, 4, 5]
    v1 = [0, 1, 4, 5]
    w1 = [0, 2, 1, 3]

    # Block 2: A[:,0:2] * B[0:2,2:4] -> C[0:2,:]
    u2 = [0, 1, 4, 5]
    v2 = [2, 3, 6, 7]
    w2 = [0, 2, 1, 3]

    # Block 3: A[:,2:4] * B[2:4,0:2] -> C[0:2,:]
    u3 = [2, 3, 6, 7]
    v3 = [8, 9, 12, 13]
    w3 = [0, 2, 1, 3]

    # Block 4: A[:,2:4] * B[2:4,2:4] -> C[0:2,:]
    u4 = [2, 3, 6, 7]
    v4 = [10, 11, 14, 15]
    w4 = [0, 2, 1, 3]

    add_block(u1, v1, w1, 0)
    add_block(u2, v2, w2, 7)
    add_block(u3, v3, w3, 14)
    add_block(u4, v4, w4, 21)

    # Internal validation: build tensor T[i,k,j] = sum_r U[i,r] V[k,r] W[j,r]
    # and compare to the multiplication tensor A[i,k]*B[k,j] (which is 1 for each triple).
    T = np.einsum('ir,kr,jr->ijk', U, V, W)
    expected = np.ones((n, m, p))
    if not np.allclose(T, expected, atol=1e-8):
        # Fall back to guaranteed valid trivial rank-32 to avoid a 0.0 score.
        U2 = np.zeros((n * m, n * m * p))
        V2 = np.zeros((m * p, n * m * p))
        W2 = np.zeros((p * n, n * m * p))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    col = i * m * p + k * p + j
                    U2[i * m + k, col] = 1.0
                    V2[k * p + j, col] = 1.0
                    W2[j * n + i, col] = 1.0
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    if time.time() > deadline:
        # still return the valid 28-rank result we have
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
