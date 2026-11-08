import time
import numpy as np

DIMENSIONS = [2, 4, 6]
DEADLINE = 140.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1; W[3][1] -= 1
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    U[2][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    U[1][6] += 1; U[3][6] -= 1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[3][6] += 1
    return U, V, W

def _trivial_123():
    U = [[0.0]*6 for _ in range(2)]
    V = [[0.0]*6 for _ in range(6)]
    W = [[0.0]*6 for _ in range(3)]
    for j in range(2):
        for k in range(3):
            col = j*3 + k
            U[j][col] = 1.0
            V[j*3 + k][col] = 1.0
            W[k][col] = 1.0
    return U, V, W

def _kron_decompose(U1, V1, W1, U2, V2, W2):
    r1 = len(U1[0])
    r2 = len(U2[0])
    n1, m1, p1 = len(U1), len(V1), len(W1)
    n2, m2, p2 = len(U2), len(V2), len(W2)
    
    U = [[0.0]*(r1*r2) for _ in range(n1*n2)]
    V = [[0.0]*(r1*r2) for _ in range(m1*m2)]
    W = [[0.0]*(r1*r2) for _ in range(p1*p2)]
    
    for l in range(r1):
        for m_idx in range(r2):
            col = l*r2 + m_idx
            for i1 in range(n1):
                for i2 in range(n2):
                    row = i1*n2 + i2
                    U[row][col] = U1[i1][l] * U2[i2][m_idx]
            for j1 in range(m1):
                for j2 in range(m2):
                    row = j1*m2 + j2
                    V[row][col] = V1[j1][l] * V2[j2][m_idx]
            for k1 in range(p1):
                for k2 in range(p2):
                    row = k1*p2 + k2
                    W[row][col] = W1[k1][l] * W2[k2][m_idx]
    return U, V, W

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    U1, V1, W1 = _strassen_222()
    U2, V2, W2 = _trivial_123()
    U, V, W = _kron_decompose(U1, V1, W1, U2, V2, W2)
    
    if time.time() - start > DEADLINE - 5:
        return {"U": U, "V": V, "W": W}
    
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_computed = np.zeros((n, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                C_computed[i, k] += val
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
