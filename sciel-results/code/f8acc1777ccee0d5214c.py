"""Tensor decomposition of 5x5x5 matrix multiplication."""
import time
import random

def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = m = p = 5
    U_cols = []
    V_cols = []
    W_cols = []

    # Frame terms: rows/cols where i==4 or j==4, all k
    for i in range(n):
        for j in range(m):
            if i == 4 or j == 4:
                for k in range(p):
                    U_cols.append({i * m + j: 1})
                    V_cols.append({j * p + k: 1})
                    W_cols.append({k * n + i: 1})

    # Boundary terms: i,j in 0..3, k==4
    for i in range(4):
        for j in range(4):
            U_cols.append({i * m + j: 1})
            V_cols.append({j * p + 4: 1})
            W_cols.append({4 * n + i: 1})

    # 4x4x4 core: use Strassen on 2x2x2 sub-blocks (4 blocks, rank 7 each)
    for ib in range(2):
        for jb in range(2):
            for kb in range(2):
                # i indices: 2*ib, 2*ib+1
                # j indices: 2*jb, 2*jb+1
                # k indices: 2*kb, 2*kb+1
                i0, i1 = 2*ib, 2*ib+1
                j0, j1 = 2*jb, 2*jb+1
                k0, k1 = 2*kb, 2*kb+1
                
                # M1 = (a+d)(e+h)
                U_cols.append({i0*m+j0: 1, i1*m+j1: 1})
                V_cols.append({j0*p+k0: 1, j1*p+k1: 1})
                W_cols.append({k0*n+i0: 1, k1*n+i1: 1})
                
                # M2 = (c+d)e
                U_cols.append({i1*m+j0: 1, i1*m+j1: 1})
                V_cols.append({j0*p+k0: 1})
                W_cols.append({k0*n+i1: 1})
                
                # M3 = a(f-h)
                U_cols.append({i0*m+j0: 1})
                V_cols.append({j0*p+k1: 1, j1*p+k1: -1})
                W_cols.append({k1*n+i0: 1})
                
                # M4 = d(g-c)
                U_cols.append({i1*m+j1: 1})
                V_cols.append({j1*p+k0: 1, j0*p+k0: -1})
                W_cols.append({k0*n+i0: 1, k0*n+i1: 1})
                
                # M5 = (a+b)h
                U_cols.append({i0*m+j0: 1, i0*m+j1: 1})
                V_cols.append({j1*p+k1: 1})
                W_cols.append({k0*n+i0: -1, k1*n+i0: 1})
                
                # M6 = (c-a)(e+f)
                U_cols.append({i1*m+j0: 1, i0*m+j0: -1})
                V_cols.append({j0*p+k0: 1, j0*p+k1: 1})
                W_cols.append({k1*n+i1: 1})
                
                # M7 = (b-d)(g+h)
                U_cols.append({i0*m+j1: 1, i1*m+j1: -1})
                V_cols.append({j1*p+k0: 1, j1*p+k1: 1})
                W_cols.append({k0*n+i0: 1, k1*n+i1: 1})

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify with random tests
    rng = random.Random(12345)
    for trial in range(100):
        if time.time() > tmax:
            break
        A = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(0, 5) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_test[i][k] = s
        if C_test != C_true:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
