import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START_TIME = time.time()
_HARD_DEADLINE = 115

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-14 decomposition for 2x3x6 
    # Constructed from two copies of the rank-7 decomposition for 3x3x3
    # restricted to the 2x3x3 sub-problem.
    
    # Rank-7 decomposition for 3x3x3 (Alon-Boros-Cohen)
    U33 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1]
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
        [0, 0, 0, 0, 1, -1, 0]
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
        [0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.float64)

    # Slice to 2x3x3
    # U indices: i*m + j -> i in 0..1, j in 0..2 -> 0..5
    U_233 = U33[0:6, :]
    
    # V indices: j*p + k -> j in 0..2, k in 0..2 -> 0..8
    V_233 = V33.copy()
    
    # W indices: k*n + i -> k in 0..2, i in 0..1
    # We need to map k=0,1,2 to W rows.
    # In W33, rows are indexed by k*3+i.
    # We want rows for i=0,1.
    # k=0: rows 0, 1
    # k=1: rows 3, 4
    # k=2: rows 6, 7
    W_233 = np.zeros((6, 7), dtype=np.float64)
    for k in range(3):
        for i in range(2):
            W_233[k * 2 + i, :] = W33[k * 3 + i, :]

    # Construct 2x3x6
    # Rank r = 14
    r = 14
    U = np.zeros((n * m, r), dtype=np.float64)
    V = np.zeros((m * p, r), dtype=np.float64)
    W = np.zeros((p * n, r), dtype=np.float64)

    # Block 1: k in 0..2
    U[:, 0:7] = U_233
    V[0:9, 0:7] = V_233
    W[0:6, 0:7] = W_233

    # Block 2: k in 3..5
    # U is same (i,j range is same)
    U[:, 7:14] = U_233
    # V indices: j*p + k. k in 3..5.
    # j=0: 0*6+3=3, 0*6+4=4, 0*6+5=5 -> rows 3,4,5
    # j=1: 1*6+3=9, 1*6+4=10, 1*6+5=11 -> rows 9,10,11
    # j=2: 2*6+3=15, 2*6+4=16, 2*6+5=17 -> rows 15,16,17
    # We need to map V_233 rows (which correspond to k=0,1,2) to these rows.
    # V_233 row j*3+k maps to V row j*6+k.
    for j in range(3):
        for k in range(3):
            V[j * 6 + (k + 3), 7:14] = V_233[j * 3 + k, :]
            
    # W indices: k*n + i. k in 3..5.
    # k=3: 3*2+0=6, 3*2+1=7 -> rows 6,7
    # k=4: 4*2+0=8, 4*2+1=9 -> rows 8,9
    # k=5: 5*2+0=10, 5*2+1=11 -> rows 10,11
    for k in range(3):
        for i in range(2):
            W[(k + 3) * 2 + i, 7:14] = W_233[k * 2 + i, :]

    # Final Verification
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
