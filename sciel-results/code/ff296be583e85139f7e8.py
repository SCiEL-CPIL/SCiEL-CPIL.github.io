"""Matrix multiplication 3x4x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _build_rank44():
    n, m, p = DIMENSIONS  # 3,4,4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x rank
    V = [[0]*rank for _ in range(m*p)]  # 16 x rank
    W = [[0]*rank for _ in range(p*n)]  # 12 x rank
    col = 0

    def add(u, v, w):
        nonlocal col
        for (ij, c) in u.items():
            U[ij][col] += c
        for (jk, c) in v.items():
            V[jk][col] += c
        for (ik, c) in w.items():
            W[ik][col] += c
        col += 1

    def ui(i, j): return i*m + j
    def vi(j, k): return j*p + k
    def wi(i, k): return k*n + i  # W[k*n+i]

    # Strassen 7-term decomposition of a 2x2x2 block
    # local (i,j),(j,k),(i,k) in {0,1}
    def strassen_block(i0, i1, j0, j1, k0, k1):
        # u vectors over local (i,j)
        def Uv(a00, a01, a10, a11):
            return {ui(i0,j0): a00, ui(i0,j1): a01, ui(i1,j0): a10, ui(i1,j1): a11}
        def Vv(b00, b01, b10, b11):
            return {vi(j0,k0): b00, vi(j0,k1): b01, vi(j1,k0): b10, vi(j1,k1): b11}
        def Wv(c00, c01, c10, c11):
            return {wi(i0,k0): c00, wi(i0,k1): c01, wi(i1,k0): c10, wi(i1,k1): c11}
        terms = [
            (Uv(1,0,0,1), Vv(1,0,0,1), Wv(1,0,0,1)),   # M1
            (Uv(0,0,1,1), Vv(1,0,0,0), Wv(0,0,1,-1)),  # M2
            (Uv(1,0,0,0), Vv(0,1,0,-1), Wv(0,1,0,1)),  # M3
            (Uv(0,0,0,1), Vv(-1,0,1,0), Wv(1,0,1,0)),  # M4
            (Uv(1,1,0,0), Vv(0,0,0,1), Wv(-1,1,0,0)),  # M5
            (Uv(-1,0,1,0), Vv(1,1,0,0), Wv(0,0,0,1)),  # M6
            (Uv(0,1,0,-1), Vv(0,0,1,1), Wv(1,0,0,0)),  # M7
        ]
        for u, v, w in terms:
            add(u, v, w)

    def schoolbook_1x2x2(i, j0, j1, k0, k1):
        for j in (j0, j1):
            for k in (k0, k1):
                add({ui(i,j):1}, {vi(j,k):1}, {wi(i,k):1})

    # 4 blocks: (i-half, j-half, k-half)
    # j-halves: {0,1} and {2,3}; k-halves: {0,1} and {2,3}
    for (j0, j1) in [(0,1), (2,3)]:
        for (k0, k1) in [(0,1), (2,3)]:
            strassen_block(0, 1, j0, j1, k0, k1)   # rows 0,1 -> 7
            schoolbook_1x2x2(2, j0, j1, k0, k1)     # row 2 -> 4
    assert col == rank, col
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # T[im_j, jp_k, kn_i] should be 1
    for im_j in range(n*m):
        i, j = divmod(im_j, m)
        for jp_k in range(m*p):
            j2, k = divmod(jp_k, p)
            if j2 != j: 
                # T should be 0 here
                val = float(np.einsum('c,c,c->', Ua[im_j], Va[jp_k], Wa[k*n+i]))
                if abs(val) > 1e-9:
                    return False
                continue
            kn_i = k*n + i
            val = float(np.einsum('c,c,c->', Ua[im_j], Va[jp_k], Wa[kn_i]))
            if abs(val - 1) > 1e-9:
                return False
    return True

def construct():
    t0 = time.time()
    deadline = t0 + 120  # safety margin under 150s kill
    best = None
    try:
        cand = _build_rank44()
        if _verify(cand["U"], cand["V"], cand["W"]):
            best = cand
        else:
            # fallback schoolbook
            best = _schoolbook()
    except Exception:
        best = None
    if best is None:
        best = _schoolbook()
    if time.time() > deadline:
        pass
    return best

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
