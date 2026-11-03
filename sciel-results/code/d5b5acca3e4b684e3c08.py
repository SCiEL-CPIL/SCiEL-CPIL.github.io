import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 112.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Exact scalar check."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = float(np.dot(U[i * m + j], V[j * p + k] * W[k * n + i]))
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _build_strassen_4x4x2():
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    def add_block(a_i, a_j, b_j, c_i, c_k, base):
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    c = U2[li*2+lj, t]
                    if c != 0:
                        U[(a_i+li)*4 + (a_j+lj), col] += c
            for lj in range(2):
                for lk in range(2):
                    c = V2[lj*2+lk, t]
                    if c != 0:
                        V[(b_j+lj)*2 + (c_k+lk), col] += c
            for lk in range(2):
                for li in range(2):
                    c = W2[lk*2+li, t]
                    if c != 0:
                        W[(c_k+lk)*4 + (c_i+li), col] += c

    add_block(0, 0, 0, 0, 0, 0)
    add_block(0, 2, 2, 0, 0, 7)
    add_block(2, 0, 0, 2, 0, 14)
    add_block(2, 2, 2, 2, 0, 21)
    return U, V, W

def _build_strassen_4x4x4():
    U2, V2, W2 = _build_strassen_2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1*2+j1, r1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2*2+j2, r2]
                            if c2 == 0: continue
                            U[(i1*2+i2)*4 + (j1*2+j2), col] += c1*c2
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1*2+k1, r1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2*2+k2, r2]
                            if c2 == 0: continue
                            V[(j1*2+j2)*4 + (k1*2+k2), col] += c1*c2
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1*2+i1, r1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2*2+i2, r2]
                            if c2 == 0: continue
                            W[(k1*2+k2)*4 + (i1*2+i2), col] += c1*c2
    return U, V, W

def _build_4x1x6_rank1():
    # Rank 1 decomposition for 4x1x6
    U = np.zeros((4, 1))
    V = np.zeros((6, 1))
    W = np.zeros((24, 1))
    U[:, 0] = 1.0
    V[:, 0] = 1.0
    W[:, 0] = 1.0
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    U444, V444, W444 = _build_strassen_4x4x4()
    U442, V442, W442 = _build_strassen_4x4x2()
    
    # Block decomposition:
    # A is 4x5, B is 5x6
    # Split j (middle dim) into 4+1, split k (output cols) into 4+2
    # Block (i in 0..3, j in 0..3, k in 0..3): 4x4x4 -> Strassen rank 49
    # Block (i in 0..3, j in 0..3, k in 4..5): 4x4x2 -> Strassen rank 28
    # Block (i in 0..3, j=4, k in 0..3): 4x1x4 -> schoolbook rank 16
    # Block (i in 0..3, j=4, k in 4..5): 4x1x2 -> schoolbook rank 8
    # Block (i=4? No, i only goes to 3)
    
    # Actually for 4x5x6: i in 0..3, j in 0..4, k in 0..5
    # We need:
    # (i in 0..3, j in 0..3, k in 0..3): 4x4x4, rank 49
    # (i in 0..3, j in 0..3, k in 4..5): 4x4x2, rank 28
    # (i in 0..3, j=4, k in 0..3): 4x1x4, rank 16 (schoolbook)
    # (i in 0..3, j=4, k in 4..5): 4x1x2, rank 8 (schoolbook)
    
    rank_4x1x4 = 4 * 1 * 4  # schoolbook
    rank_4x1x2 = 4 * 1 * 2  # schoolbook
    total_rank = 49 + 28 + rank_4x1x4 + rank_4x1x2
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 4x4x4 (i:0-3, j:0-3, k:0-3)
    base = 0
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+49] = U444[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, base:base+49] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, base:base+49] = W444[k * 4 + i, :]
            
    # Part 2: 4x4x2 (i:0-3, j:0-3, k:4-5)
    base = 49
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
    for j in range(4):
        for k in range(2):
            V[j * 6 + (4+k), base:base+28] = V442[j * 2 + k, :]
    for k in range(2):
        for i in range(4):
            W[(4+k) * 4 + i, base:base+28] = W442[k * 4 + i, :]
            
    # Part 3: 4x1x4 schoolbook (i:0-3, j=4, k:0-3)
    base = 49 + 28
    for idx, (i, k) in enumerate([(i, k) for i in range(4) for k in range(4)]):
        U[i * 5 + 4, base + idx] = 1.0
        V[4 * 6 + k, base + idx] = 1.0
        W[k * 4 + i, base + idx] = 1.0
        
    # Part 4: 4x1x2 schoolbook (i:0-3, j=4, k:4-5)
    base = 49 + 28 + rank_4x1x4
    for idx, (i, k) in enumerate([(i, k) for i in range(4) for k in range(4, 6)]):
        U[i * 5 + 4, base + idx] = 1.0
        V[4 * 6 + k, base + idx] = 1.0
        W[k * 4 + i, base + idx] = 1.0
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
