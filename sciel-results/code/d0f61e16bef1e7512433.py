import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 6]
_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank66(n, m, p):
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    def add_schoolbook(i, j, k):
        nonlocal col
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1

    def add_strassen(i_base, j_base, k_base):
        nonlocal col
        a = i_base*m + j_base
        b = i_base*m + (j_base+1)
        c = (i_base+1)*m + j_base
        d = (i_base+1)*m + (j_base+1)
        e = j_base*p + k_base
        f = j_base*p + (k_base+1)
        g = (j_base+1)*p + k_base
        h = (j_base+1)*p + (k_base+1)
        P = k_base*n + i_base
        Q = (k_base+1)*n + i_base
        Rr = k_base*n + (i_base+1)
        S = (k_base+1)*n + (i_base+1)

        # M1 = A00 * B11
        U[a][col]=1; U[d][col]=1; V[e][col]=1; V[h][col]=1; W[P][col]=1; W[S][col]=1; col+=1
        # M2 = A10 * (B01 - B00)
        U[c][col]=1; U[d][col]=1; V[e][col]=1; V[g][col]=-1; W[Rr][col]=1; W[S][col]=-1; col+=1
        # M3 = (A00 + A01) * B10
        U[a][col]=1; U[b][col]=1; V[f][col]=1; V[h][col]=-1; W[Q][col]=1; W[S][col]=1; col+=1
        # M4 = (A10 - A00) * (B11 - B00)
        U[d][col]=1; U[a][col]=-1; V[g][col]=1; V[e][col]=-1; W[P][col]=1; W[Rr][col]=1; col+=1
        # M5 = (A00 + A10) * (B11 - B01)
        U[a][col]=1; U[c][col]=1; V[h][col]=1; V[f][col]=-1; W[P][col]=-1; W[Q][col]=1; col+=1
        # M6 = (A10 - A01) * (B10 - B11)
        U[c][col]=1; U[a][col]=-1; V[e][col]=1; V[f][col]=-1; W[S][col]=1; col+=1
        # M7 = (A01 + A11) * (B00 - B10)
        U[b][col]=1; U[d][col]=-1; V[g][col]=1; V[h][col]=-1; W[P][col]=1; col+=1

    # Row 0: Schoolbook (1x4x6) -> 24 terms
    for j in range(m):
        for k in range(p):
            add_schoolbook(0, j, k)
            
    # Rows 1-2: Strassen on 2x2x2 blocks
    for kb in range(p//2):
        for jb in range(m//2):
            add_strassen(1, jb*2, kb*2)
            
    return U, V, W

def construct():
    n, m, p = 3, 4, 6
    U, V, W = _build_rank66(n, m, p)
    return {"U": U, "V": V, "W": W}
