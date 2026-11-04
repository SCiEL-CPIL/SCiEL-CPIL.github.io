"""Strassen block decomposition for 6x6 matrix multiplication (rank 189)."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n):
    """Schoolbook rank-n^3 decomposition for n x n. Returns (U,V,W) with
    U,V,W shape (n*n, n^3). U[(i,j),c], V[(j,k),c], W[(k,i),c]."""
    rank = n * n * n
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    c = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i*n + j][c] = 1
                V[j*n + k][c] = 1
                W[k*n + i][c] = 1
                c += 1
    return U, V, W


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    # 6x6x6: split into 2x2 blocks of size 3.
    h = 3
    # Base: 3x3x3 schoolbook
    U_h, V_h, W_h = _schoolbook(h)
    r_h = len(U_h[0])  # 27

    rank = 7 * r_h
    N = 6
    U = [[0]*rank for _ in range(N*N)]
    V = [[0]*rank for _ in range(N*N)]
    W = [[0]*rank for _ in range(N*N)]

    def unit(a, b):
        return {(a, b): 1}
    def add(d1, d2):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0) + v
        return d
    def sub(d1, d2):
        d = dict(d1)
        for k, v in d2.items():
            d[k] = d.get(k, 0) - v
        return d

    A11, A12, A21, A22 = unit(0,0), unit(0,1), unit(1,0), unit(1,1)
    B11, B12, B21, B22 = unit(0,0), unit(0,1), unit(1,0), unit(1,1)

    prods = [
        (add(A11, A22), add(B11, B22), [(0,0,1)]),
        (add(A21, A22), B11, [(1,0,1),(1,1,-1)]),
        (A11, sub(B21, B22), [(0,1,1),(1,1,1)]),
        (A22, sub(B12, B11), [(0,0,1),(1,0,1)]),
        (add(A11, A12), B22, [(0,0,-1),(0,1,1)]),
        (sub(A21, A11), add(B11, B12), [(1,1,1)]),
        (sub(A12, A22), add(B21, B22), [(0,0,1),(1,1,1)]),
    ]

    for p_idx, (a_combo, b_combo, c_targets) in enumerate(prods):
        base = p_idx * r_h
        # U: block (a,b) of A. A row-block a, col-block b.
        for (a, b), ca in a_combo.items():
            if ca == 0:
                continue
            for i in range(h):
                for k in range(h):
                    u_h_row = i * h + k
                    u_row = (a*h + i) * N + (b*h + k)
                    for c in range(r_h):
                        U[u_row][base + c] += ca * U_h[u_h_row][c]
        # V: block (j,k) of B.
        for (j, k), cb in b_combo.items():
            if cb == 0:
                continue
            for jloc in range(h):
                for kloc in range(h):
                    v_h_row = jloc * h + kloc
                    v_row = (j*h + jloc) * N + (k*h + kloc)
                    for c in range(r_h):
                        V[v_row][base + c] += cb * V_h[v_h_row][c]
        # W: C(ci, cj) -> i_block=ci, k_block=cj. W[(k,i),r].
        for (ci, cj, cc) in c_targets:
            for iloc in range(h):
                for kloc in range(h):
                    w_h_row = kloc * h + iloc
                    w_row = (cj*h + kloc) * N + (ci*h + iloc)
                    for c in range(r_h):
                        W[w_row][base + c] += cc * W_h[w_h_row][c]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
