"""Matrix multiplication 4x4x6 tensor decomposition (rank 36 via Strassen blocks)."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]
DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6

    # Strassen 2x2x2 rank-7 terms: (a, b, c) lists of (idx1, idx2, coef).
    # a: (I,J) for A, b: (J,K) for B, c: (I,K) for C.
    terms = [
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1)]),          # M1
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1)]),           # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)]),            # M3
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,1,1),(0,0,-1)]),  # M4
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # M5
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),   # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),   # M7
    ]

    # Inner 2x2x2 schoolbook (rank 4): Us,Vs,Ws shape (4,4).
    Us = np.zeros((4, 4)); Vs = np.zeros((4, 4)); Ws = np.zeros((4, 4))
    c = 0
    for li in range(2):
        for lj in range(2):
            for lk in range(2):
                Us[li*2+lj, c] = 1; Vs[lj*2+lk, c] = 1; Ws[lk*2+li, c] = 1
                c += 1

    # Build 4x4x4 (rank 28).
    rank444 = 28
    U444 = np.zeros((16, rank444)); V444 = np.zeros((16, rank444)); W444 = np.zeros((16, rank444))
    for cb in range(7):
        a, b, cc = terms[cb]
        for ci in range(4):
            g = cb*4 + ci
            for (I,J,va) in a:
                for li in range(2):
                    for lj in range(2):
                        uv = Us[li*2+lj, ci]
                        if uv: U444[2*I+li, 2*J+lj, g] += va*uv
            for (J,K,vb) in b:
                for lj in range(2):
                    for lk in range(2):
                        vv = Vs[lj*2+lk, ci]
                        if vv: V444[2*J+lj, 2*K+lk, g] += vb*vv
            for (I,K,vw) in cc:
                for lk in range(2):
                    for li in range(2):
                        wv = Ws[lk*2+li, ci]
                        if wv: W444[2*K+lk, 2*I+li, g] += vw*wv

    # Build 4x4x2 schoolbook (rank 8).
    rank442 = 8
    U442 = np.zeros((16, rank442)); V442 = np.zeros((8, rank442)); W442 = np.zeros((8, rank442))
    c = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U442[i*4+j, c] = 1; V442[j*2+k, c] = 1; W442[k*4+i, c] = 1
                c += 1

    # Combine to 4x4x6 (rank 36).
    total = rank444 + rank442
    U446 = np.zeros((16, total)); V446 = np.zeros((24, total)); W446 = np.zeros((24, total))
    U446[:, :rank444] = U444
    for j in range(4):
        for k in range(4):
            V446[j*6+k, :rank444] = V444[j*4+k, :rank444]
    for k in range(4):
        for i in range(4):
            W446[k*4+i, :rank444] = W444[k*4+i, :rank444]
    U446[:, rank444:] = U442
    for j in range(4):
        for k in range(2):
            V446[j*6+4+k, rank444:] = V442[j*2+k, :]
    for k in range(2):
        for i in range(4):
            W446[(4+k)*4+i, rank444:] = W442[k*4+i, :]

    def _verify(U, V, W, n, m, p):
        Ua = np.array(U).reshape(n*m, -1)
        Va = np.array(V).reshape(m*p, -1)
        Wa = np.array(W).reshape(p*n, -1)
        rank = Ua.shape[1]
        res = np.zeros((n, m, p))
        for c in range(rank):
            u = Ua[:, c].reshape(n, m)
            v = Va[:, c].reshape(m, p)
            w = Wa[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        return np.allclose(res, np.ones((n, m, p)), atol=1e-6)

    if not _verify(U446, V446, W446, 4, 4, 6):
        # Fallback schoolbook rank 96.
        rank = 96
        U = np.zeros((16, rank)); V = np.zeros((24, rank)); W = np.zeros((24, rank))
        c = 0
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    U[i*4+j, c] = 1; V[j*6+k, c] = 1; W[k*4+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U446.tolist(), "V": V446.tolist(), "W": W446.tolist()}
# EVOLVE-BLOCK-END
