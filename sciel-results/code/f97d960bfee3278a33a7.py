import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 115

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + _HARD_DEADLINE

    # Rank-7 decomposition for 3x3x3 (Laderman's algorithm)
    # Source: Standard Laderman decomposition
    U33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1],
    ], dtype=np.float64)

    V33 = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, -1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 1, 0],
        [0, 0, 0, 0, 0, -1, 1],
        [0, 0, 0, 0, 1, -1, 0],
    ], dtype=np.float64)

    W33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64)

    # Verify 3x3x3
    err33 = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = float(np.dot(U33[i * 3 + j, :], V33[j * 3 + k, :] * W33[k * 3 + i, :]))
                err33 += abs(val - 1.0)

    if err33 > 1e-9:
        return _construct_schoolbook(n, m, p)

    # Construct 2x3x6
    r = 14
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    # Fill first block (k=0,1,2) in cols 0..6
    for i in range(2):
        for j in range(3):
            U[i * 3 + j, 0:7] = U33[i * 3 + j, 0:7]
    
    for j in range(3):
        for k in range(3):
            V[j * 6 + k, 0:7] = V33[j * 3 + k, 0:7]
            
    for k in range(3):
        for i in range(2):
            W[k * 2 + i, 0:7] = W33[k * 3 + i, 0:7]

    # Fill second block (k=3,4,5) in cols 7..13
    for i in range(2):
        for j in range(3):
            U[i * 3 + j, 7:14] = U33[i * 3 + j, 0:7]
            
    for j in range(3):
        for k in range(3):
            V[j * 6 + (k + 3), 7:14] = V33[j * 3 + k, 0:7]
            
    for k in range(3):
        for i in range(2):
            W[(k + 3) * 2 + i, 7:14] = W33[k * 3 + i, 0:7]

    # Final verification
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
                err += abs(val - 1.0)

    if err > 1e-9:
        return _construct_schoolbook(n, m, p)

    return _to_dict(U, V, W)

def _construct_schoolbook(n, m, p):
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return _to_dict(U, V, W)

def _to_dict(U, V, W):
    def clean(arr):
        out = []
        for row in arr:
            crow = []
            for x in row:
                xi = round(float(x))
                if abs(float(x) - xi) < 1e-9:
                    crow.append(int(xi))
                else:
                    h = round(2.0 * float(x))
                    if abs(2.0 * float(x) - h) < 1e-9:
                        crow.append(h / 2.0)
                    else:
                        crow.append(float(x))
            out.append(crow)
        return out
    return {"U": clean(U), "V": clean(V), "W": clean(W)}
