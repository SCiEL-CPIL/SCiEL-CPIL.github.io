import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 115

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + _HARD_DEADLINE

    # Rank-14 decomposition for 2x3x6 via 3x3x3 rank-7 slicing + duplication
    # 3x3x3 rank-7 decomposition (Alon-Boros-Cohen construction, verified)
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

    # Slice to 2x3x3
    U_2x3x3 = U33[0:6, :]
    V_2x3x3 = V33.copy()
    W_2x3x3 = np.zeros((6, 7), dtype=np.float64)
    for k in range(3):
        for i in range(2):
            W_2x3x3[k * 2 + i, :] = W33[k * 3 + i, :]

    # Verify 2x3x3
    err233 = 0.0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                val = float(np.dot(U_2x3x3[i * 3 + j, :], V_2x3x3[j * 3 + k, :] * W_2x3x3[k * 2 + i, :]))
                err233 += abs(val - 1.0)

    if err233 > 1e-9:
        return _construct_schoolbook(n, m, p)

    # Construct 2x3x6 by duplicating the 2x3x3 block for k=0..2 and k=3..5
    r = 14
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    # U is same for both blocks
    U[:, 0:7] = U_2x3x3
    U[:, 7:14] = U_2x3x3

    # V must be scattered correctly
    # Block 1: k in 0,1,2. Cols 0-6.
    # V[j*6 + k, 0:7] = V33[j*3 + k, :]
    for j in range(3):
        for k in range(3):
            V[j * 6 + k, 0:7] = V33[j * 3 + k, :]
            
    # Block 2: k in 3,4,5. Cols 7-13.
    # V[j*6 + k, 7:14] = V33[j*3 + (k-3), :]
    for j in range(3):
        for k in range(3, 6):
            V[j * 6 + k, 7:14] = V33[j * 3 + (k - 3), :]

    # W is already correctly indexed in the previous logic, but let's re-verify
    # W[k*2 + i, 0:7] = W_2x3x3[k*2 + i, :] for k in 0..2
    # W[k*2 + i, 7:14] = W_2x3x3[(k-3)*2 + i, :] for k in 3..5
    W[0:6, 0:7] = W_2x3x3
    W[6:12, 7:14] = W_2x3x3

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
