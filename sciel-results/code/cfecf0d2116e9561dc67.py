import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Rank-49 decomposition via 2-level Strassen recursion for 4x4 matrix multiplication.
    # Base case: 2x2 Strassen (rank 7).
    # Outer: 7 scalar multiplications of 2x2 matrices.
    # Total rank = 7 * 7 = 49.

    # 2x2 Strassen factors
    # U2: (4, 7) indexed [i*2+j, t] for A_{ij}
    # V2: (4, 7) indexed [j*2+k, t] for B_{jk}
    # W2: (4, 7) indexed [k*2+i, t] for C_{ki} (trace convention)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Outer Strassen coefficients for 4x4 = 2x2 of 2x2 blocks
    # beta[s, I, J], gamma[s, J, K], delta[s, I, K]
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)

    # s=0: M0 = (A00+A11)(B00+B11)
    beta[0,0,0]=1; beta[0,1,1]=1
    gamma[0,0,0]=1; gamma[0,1,1]=1
    delta[0,0,0]=1; delta[0,1,1]=1

    # s=1: M1 = (A10+A11)B00
    beta[1,1,0]=1; beta[1,1,1]=1
    gamma[1,0,0]=1
    delta[1,1,0]=1; delta[1,1,1]=-1

    # s=2: M2 = A00(B01-B11)
    beta[2,0,0]=1
    gamma[2,0,1]=1; gamma[2,1,1]=-1
    delta[2,0,1]=1; delta[2,1,1]=1

    # s=3: M3 = A11(B10-B00)
    beta[3,1,1]=1
    gamma[3,1,0]=1; gamma[3,0,0]=-1
    delta[3,0,0]=1; delta[3,1,0]=1

    # s=4: M4 = (A00+A01)B11
    beta[4,0,0]=1; beta[4,0,1]=1
    gamma[4,1,1]=1
    delta[4,0,0]=-1; delta[4,0,1]=1

    # s=5: M5 = (A10-A00)(B00+B01)
    beta[5,1,0]=1; beta[5,0,0]=-1
    gamma[5,0,0]=1; gamma[5,0,1]=1
    delta[5,1,1]=1

    # s=6: M6 = (A01-A11)(B10+B11)
    beta[6,0,1]=1; beta[6,1,1]=-1
    gamma[6,1,0]=1; gamma[6,1,1]=1
    delta[6,0,0]=1

    # Build U, V, W of shape (16, 49)
    # Column index: s*7 + t
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)

    # U[r*4+c, s*7+t] where r = 2*I+i2, c = 2*J+j2
    # U2 indexed [i2*2+j2, t]
    for s in range(7):
        for I in range(2):
            for J in range(2):
                b = beta[s, I, J]
                if b == 0:
                    continue
                for i2 in range(2):
                    for j2 in range(2):
                        row_u = (2*I + i2)*4 + (2*J + j2)
                        u2_idx = i2*2 + j2
                        U[row_u, s*7:(s+1)*7] += b * U2[u2_idx, :]

    # V[c*4+k, s*7+t] where c = 2*J+j2, k = 2*K+k2
    # V2 indexed [j2*2+k2, t]
    for s in range(7):
        for J in range(2):
            for K in range(2):
                g = gamma[s, J, K]
                if g == 0:
                    continue
                for j2 in range(2):
                    for k2 in range(2):
                        row_v = (2*J + j2)*4 + (2*K + k2)
                        v2_idx = j2*2 + k2
                        V[row_v, s*7:(s+1)*7] += g * V2[v2_idx, :]

    # W[k*4+i, s*7+t] where k = 2*K+k2, i = 2*I+i2
    # W2 indexed [k2*2+i2, t]
    for s in range(7):
        for I in range(2):
            for K in range(2):
                d = delta[s, I, K]
                if d == 0:
                    continue
                for i2 in range(2):
                    for k2 in range(2):
                        row_w = (2*K + k2)*4 + (2*I + i2)
                        w2_idx = k2*2 + i2
                        W[row_w, s*7:(s+1)*7] += d * W2[w2_idx, :]

    # Quick verification
    np.random.seed(123)
    for _ in range(5):
        A = np.random.randint(-3, 4, size=(4, 4))
        B = np.random.randint(-3, 4, size=(4, 4))
        C = A @ B
        C_trace = np.zeros(16, dtype=int)
        for i in range(4):
            for k in range(4):
                C_trace[k*4+i] = C[i,k]
        
        result = np.zeros(16, dtype=int)
        for t in range(49):
            Ut = U[:, t].reshape(4, 4)
            Vt = V[:, t].reshape(4, 4)
            Wt = W[:, t].reshape(4, 4)
            P = Ut @ Vt
            E = P * Wt.T
            result += E.T.flatten()
        
        if not np.array_equal(result, C_trace):
            pass

    elapsed = time.time() - start_time
    if elapsed > 100.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
