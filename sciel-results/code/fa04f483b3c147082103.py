import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def _check_deadline():
    return (time.time() - _start_time) > (_DEADLINE - _SAFETY_MARGIN)

def _verify(U, V, W, n=2, m=3, p=5, tol=1e-9):
    """Verify exact decomposition."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    tensor = np.zeros((n, m, p))
    for r in range(rank):
        u = Ua[:, r].reshape(n, m)
        v = Va[:, r].reshape(m, p)
        w = Wa[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i, j, k] += u[i, j] * v[j, k] * w[k, i]
    return np.allclose(tensor, np.ones((n, m, p)), atol=tol)

def _naive(n=2, m=3, p=5):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2(U, V, W, r, i_base, j_base, k_base, n=2, m=2, p=2):
    """Add 7 Strassen terms for 2x2x2 block starting at indices."""
    # U indices: (i_base + di)*m + (j_base + dj)
    # V indices: (j_base + dj)*p + (k_base + dk)
    # W indices: (k_base + dk)*n + (i_base + di)
    # Standard Strassen for 2x2x2:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    
    # m1: (a11+a22)*(b11+b22)
    U[i_base * m + j_base][r] = 1
    U[(i_base+1) * m + (j_base+1)][r] = 1
    V[j_base * p + k_base][r] = 1
    V[(j_base+1) * p + (k_base+1)][r] = 1
    W[k_base * n + i_base][r] = 1
    W[(k_base+1) * n + (i_base+1)][r] = 1
    r += 1
    
    # m2: (a21+a22)*b11
    U[(i_base+1) * m + j_base][r] = 1
    U[(i_base+1) * m + (j_base+1)][r] = 1
    V[j_base * p + k_base][r] = 1
    W[(k_base+1) * n + (i_base+1)][r] = 1
    W[k_base * n + i_base][r] = -1
    r += 1
    
    # m3: a11*(b12-b22)
    U[i_base * m + j_base][r] = 1
    V[(j_base+1) * p + (k_base+1)][r] = 1
    V[j_base * p + (k_base+1)][r] = -1
    W[(k_base+1) * n + i_base][r] = 1
    W[k_base * n + (i_base+1)][r] = 1
    r += 1
    
    # m4: a22*(b21-b11)
    U[(i_base+1) * m + (j_base+1)][r] = 1
    V[(j_base+1) * p + k_base][r] = 1
    V[j_base * p + k_base][r] = -1
    W[k_base * n + (i_base+1)][r] = 1
    W[(k_base+1) * n + (i_base+1)][r] = 1
    r += 1
    
    # m5: (a11+a12)*b22
    U[i_base * m + j_base][r] = 1
    U[i_base * m + (j_base+1)][r] = 1
    V[(j_base+1) * p + (k_base+1)][r] = 1
    W[k_base * n + i_base][r] = -1
    W[(k_base+1) * n + i_base][r] = 1
    r += 1
    
    # m6: (a21-a11)*(b11+b12)
    U[(i_base+1) * m + j_base][r] = 1
    U[i_base * m + j_base][r] = -1
    V[j_base * p + k_base][r] = 1
    V[j_base * p + (k_base+1)][r] = 1
    W[(k_base+1) * n + (i_base+1)][r] = 1
    r += 1
    
    # m7: (a12-a22)*(b21+b22)
    U[i_base * m + (j_base+1)][r] = 1
    U[(i_base+1) * m + (j_base+1)][r] = -1
    V[(j_base+1) * p + k_base][r] = 1
    V[(j_base+1) * p + (k_base+1)][r] = 1
    W[k_base * n + i_base][r] = 1
    r += 1
    
    return r

def construct():
    if _check_deadline():
        n, m, p = DIMENSIONS
        return _naive(n, m, p)
    
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # Part 3: Strassen for A1 B11 (j=0,1, k=0,1) - 7 terms
    r = _strassen_2x2x2(U, V, W, r, i_base=0, j_base=0, k_base=0, n=2, m=3, p=5)
    # Part 4: Strassen for A1 B12 (j=0,1, k=2,3) - 7 terms
    r = _strassen_2x2x2(U, V, W, r, i_base=0, j_base=0, k_base=2, n=2, m=3, p=5)
    
    if not _verify(U, V, W):
        return _naive(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
