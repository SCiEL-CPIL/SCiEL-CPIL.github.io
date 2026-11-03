"""Rank-reduced decomposition for 2x5x6 matrix multiplication.

Uses Strassen (rank 7) for the 2x2x2 block and schoolbook for the
remaining rectangular blocks. Total rank = 7 + 12 + 12 + 18 = 49.

Convention: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c] with
n=2, m=5, p=6. The product is sum_c U[:,c] * V[:,c] * W[:,c]
reconstructed as C[i,k] = sum_{j,c} U[i*5+j,c] * V[j*6+k,c] * W[k*2+i,c].
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Check exact decomposition on random integer inputs (mod-free, exact ints)."""
    rng = np.random.default_rng(12345)
    for _ in range(3):
        A = rng.integers(-3, 4, size=(n, m))
        B = rng.integers(-3, 4, size=(m, p))
        C_true = A @ B
        C = np.zeros((n, p), dtype=object)
        Uc = np.array(U, dtype=object)
        Vc = np.array(V, dtype=object)
        Wc = np.array(W, dtype=object)
        for c in range(len(W[0])):
            # term = (U[:,c] outer V[:,c]) contracted with W[:,c]
            # C[i,k] += sum_j U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += Uc[i*m+j, c] * Vc[j*p+k, c] * Wc[k*n+i, c]
                    C[i, k] = C[i, k] + s
        if not np.array_equal(C, C_true):
            return False
    return True

def _build():
    n, m, p = 2, 5, 6
    # Blocks:
    #  A = [A1 (2x2) | A2 (2x3)]
    #  B = [B1 (5x2) ; B2 (5x3)]  -> B1 = [B1t (2x2); B1b (3x2)], B2 = [B2t (2x3); B2b (3x3)]
    #
    # C[:, 0:2] = A1*B1t + A2*B1b
    # C[:, 2:5] = A1*B2t + A2*B2b
    #
    # Block (A1,B1t): 2x2x2 -> Strassen rank 7, k range [0,2)
    # Block (A2,B1b): 2x3x2 -> schoolbook rank 12, k range [2,4)
    # Block (A1,B2t): 2x2x3 -> schoolbook rank 12, k range [4,7)
    # Block (A2,B2b): 2x3x3 -> schoolbook rank 18, k range [7,10)
    rank = 7 + 12 + 12 + 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # ---- Strassen 2x2x2 for A1 (rows 0,1; mid 0,1) x B1t (mid 0,1; cols 0,1) ----
    # A1[i][j] for i in 0..1, j in 0..1 -> U index i*5 + j
    # B1t[j][k] for j in 0..1, k in 0..1 -> V index j*6 + k
    # W for output C[i][k] -> W index k*2 + i
    # Strassen terms (standard):
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B10)
    # M6 = (A01-A11)(B01+B11)
    # C00 = M0+M3-M5+M6
    # C01 = M2+M4+M5
    # C10 = M1+M4-M3-M6
    # C11 = M0+M2-M1+M3
    A1_idx = lambda i, j: i * 5 + j
    B1t_idx = lambda j, k: j * 6 + k
    W_idx = lambda k, i: k * 2 + i

    strassen = [
        # (U terms [(i,j,coef)], V terms [(j,k,coef)], W terms [(k,i,coef)])
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(0,1,1),(1,0,-1),(1,1,1)]),   # M0
        ([(1,0,1),(1,1,1)], [(0,0,1)],          [(0,1,1),(1,0,1),(1,1,-1)]),            # M1
        ([(0,0,1)],          [(0,1,1),(1,1,-1)],[(0,1,1),(1,1,1)]),                     # M2
        ([(1,1,1)],          [(1,0,1),(0,0,-1)],[(0,0,-1),(1,0,-1),(1,1,1)]),           # M3
        ([(0,0,1),(0,1,1)], [(1,1,1)],          [(0,1,1),(1,0,1)]),                     # M4
        ([(1,0,1),(0,0,-1)],[(0,0,1),(1,0,1)],  [(0,1,1),(1,0,-1)]),                    # M5
        ([(0,1,1),(1,1,-1)],[(0,1,1),(1,1,1)],  [(0,1,1),(1,0,-1)]),                    # M6
    ]
    for (ut, vt, wt) in strassen:
        for (i, j, c) in ut:
            U[A1_idx(i, j)][col] += c
        for (j, k, c) in vt:
            V[B1t_idx(j, k)][col] += c
        for (k, i, c) in wt:
            W[W_idx(k, i)][col] += c
        col += 1

    # ---- Schoolbook 2x3x2 for A2 (rows 0,1; mid 2,3,4) x B1b (mid 2,3,4; cols 0,1) ----
    # A2[i][j] -> U index i*5 + (j+2), j in 0..2 (mid 2,3,4)
    # B1b[j][k] -> V index (j+2)*6 + k, j in 0..2, k in 0..1
    # W[k][i] -> W index k*2 + i
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i * 5 + (j + 2)][col] = 1
                V[(j + 2) * 6 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1

    # ---- Schoolbook 2x2x3 for A1 (rows 0,1; mid 0,1) x B2t (mid 0,1; cols 2,3,4) ----
    # A1[i][j] -> U index i*5 + j, j in 0..1
    # B2t[j][k] -> V index j*6 + (k+2), j in 0..1, k in 0..2 (cols 2,3,4)
    # W[k][i] -> W index (k+2)*2 + i
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * 5 + j][col] = 1
                V[j * 6 + (k + 2)][col] = 1
                W[(k + 2) * 2 + i][col] = 1
                col += 1

    # ---- Schoolbook 2x3x3 for A2 (rows 0,1; mid 2,3,4) x B2b (mid 2,3,4; cols 2,3,4) ----
    # A2[i][j] -> U index i*5 + (j+2)
    # B2b[j][k] -> V index (j+2)*6 + (k+2)
    # W[k][i] -> W index (k+2)*2 + i
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i * 5 + (j + 2)][col] = 1
                V[(j + 2) * 6 + (k + 2)][col] = 1
                W[(k + 2) * 2 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 110.0
    n, m, p = DIMENSIONS
    result = _build()
    # Exact verification (small, fast)
    try:
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    # Fallback: schoolbook (always valid)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
