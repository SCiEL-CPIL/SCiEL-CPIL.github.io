import numpy as np

def _strassen_2x2x2():
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def _tensor_product_decomp(U1, V1, W1):
    n, m, p = 4, 4, 4
    r1 = U1.shape[1]
    r = r1 * r1
    
    U = np.zeros((n*m, r), dtype=float)
    V = np.zeros((m*p, r), dtype=float)
    W = np.zeros((p*n, r), dtype=float)
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    idx = i * 4 + j
                    for c1 in range(r1):
                        for c2 in range(r1):
                            c = c1 * r1 + c2
                            U[idx, c] = U1[i1*2+j1, c1] * U1[i2*2+j2, c2]
    
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    idx = j * 4 + k
                    for c1 in range(r1):
                        for c2 in range(r1):
                            c = c1 * r1 + c2
                            V[idx, c] = V1[j1*2+k1, c1] * V1[j2*2+k2, c2]
    
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    idx = k * 4 + i
                    for c1 in range(r1):
                        for c2 in range(r1):
                            c = c1 * r1 + c2
                            W[idx, c] = W1[k1*2+i1, c1] * W1[k2*2+i2, c2]
    
    return U, V, W

def _validate(U, V, W, n, m, p):
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-10):
                    return False
    return True

def construct():
    n, m, p = 4, 4, 8
    
    U2, V2, W2 = _strassen_2x2x2()
    U4, V4, W4 = _tensor_product_decomp(U2, V2, W2)
    
    r4 = U4.shape[1]
    r = 2 * r4
    
    U = np.zeros((16, r), dtype=float)
    V = np.zeros((32, r), dtype=float)
    W = np.zeros((32, r), dtype=float)
    
    U[:, 0:r4] = U4
    U[:, r4:2*r4] = U4
    
    V[0:16, 0:r4] = V4
    V[16:32, r4:2*r4] = V4
    
    W[0:16, 0:r4] = W4
    W[16:32, r4:2*r4] = W4
    
    if not _validate(U, V, W, n, m, p):
        raise ValueError("Decomposition is invalid")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
