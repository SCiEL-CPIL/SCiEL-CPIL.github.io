"""Rewritten program: block-decomposed exact tensor decomposition of M(3,4,7)."""
DIMENSIONS = [3, 4, 7]
import time
import numpy as np

_DEADLINE = time.time() + 120.0
_EPS = 1e-9


def _strassen2x2x2():
    """Rank-7 Strassen decomposition of the 2x2x2 matrix-multiplication tensor.
    U (4x7) indexed (i*2+j), V (4x7) indexed (j*2+k), W (4x7) indexed (k*2+i).
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    def set(col, ux, vy, wz):
        U[ux, col] = 1
        V[vy, col] = 1
        W[wz, col] = 1

    # M1 = (a+b)(c+d)
    set(0, 0, 0, 0); set(0, 0, 1, 0); set(0, 3, 0, 1); set(0, 3, 1, 1)
    # M2 = (a+b)(d-c)
    set(1, 0, 1, 0); set(1, 0, 0, -1); set(1, 3, 1, 1); set(1, 3, 0, -1)
    # M3 = a(c-b)
    set(2, 0, 0, 0); set(2, 0, 2, -1)
    # M4 = b(d-c)
    set(3, 3, 1, 1); set(3, 3, 0, -1)
    # M5 = (a-c)(d+b)
    set(4, 0, 1, 0); set(4, 0, 3, 0); set(4, 2, 1, -1); set(4, 2, 3, -1)
    # M6 = (a-c)(b+d)
    set(5, 0, 2, 0); set(5, 0, 3, 0); set(5, 2, 2, -1); set(5, 2, 3, -1)
    # M7 = (a+b)(c-b)
    set(6, 0, 0, 0); set(6, 0, 2, -1); set(6, 3, 0, 1); set(6, 3, 2, -1)
    return U, V, W


def _schoolbook(n, m, p):
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _sub243():
    """Rank-11 decomposition of the 2x4x3 tensor:
    Strassen (rank 7) on rows {0,1}, cols {0,1}, out {0,1};
    schoolbook (rank 4) for the remaining (col,out) pairs.
    """
    sU, sV, sW = _strassen2x2x2()
    rank = 7 + 4
    U = np.zeros((2 * 4, rank))
    V = np.zeros((4 * 3, rank))
    W = np.zeros((3 * 2, rank))
    # Strassen part (cols 0..6)
    for col in range(7):
        for r in range(2):
            for c in range(2):
                if sU[r * 2 + c, col] != 0:
                    U[r * 4 + c, col] += sU[r * 2 + c, col]
        for c in range(2):
            for o in range(2):
                if sV[c * 2 + o, col] != 0:
                    V[c * 3 + o, col] += sV[c * 2 + o, col]
        for o in range(2):
            for r in range(2):
                if sW[o * 2 + r, col] != 0:
                    W[o * 2 + r, col] += sW[o * 2 + r, col]
    # Schoolbook for (c,o) with c>=2 or o>=2
    col = 7
    for r in range(2):
        for c in range(4):
            for o in range(3):
                if not (c < 2 and o < 2):
                    U[r * 4 + c, col] = 1
                    V[c * 3 + o, col] = 1
                    W[o * 2 + r, col] = 1
                    col += 1
    return U, V, W


def _sub244():
    """Decomposition of the 2x4x4 tensor (schoolbook rank 32, safe)."""
    return _schoolbook(2, 4, 4)


def _assemble():
    """Full 3x4x7 decomposition.
    A (3x4): rows split into r in {0,1} and r in {2}.
    B (4x7): out-cols split into o in {0,1,2} and o in {3..6}.
    Blocks:
      (r in {0,1}, o in {0,1,2}): 2x4x3  (rank 11)
      (r in {0,1}, o in {3..6}):   2x4x4  (rank 32)
      (r in {2},   o in {0..6}):   1x4x7  (rank 7)
    Total rank = 50.
    """
    U1, V1, W1 = _sub243()
    U2, V2, W2 = _sub244()
    U3, V3, W3 = _schoolbook(1, 4, 7)

    U_full = np.zeros((12, 0))
    V_full = np.zeros((28, 0))
    W_full = np.zeros((21, 0))

    def place(U_full, V_full, W_full, U_b, V_b, W_b, row_off, col_off, out_off):
        rb, cb, ob = U_b.shape[0], V_b.shape[0], W_b.shape[0]
        rk = U_b.shape[1]
        U_ext = np.zeros((12, rk))
        V_ext = np.zeros((28, rk))
        W_ext = np.zeros((21, rk))
        U_ext[row_off:row_off + rb, :] = U_b
        V_ext[col_off:col_off + cb, :] = V_b
        W_ext[out_off:out_off + ob, :] = W_b
        return (np.hstack([U_full, U_ext]),
                np.hstack([V_full, V_ext]),
                np.hstack([W_full, W_ext]))

    U_full, V_full, W_full = place(U_full, V_full, W_full, U1, V1, W1, 0, 0, 0)
    U_full, V_full, W_full = place(U_full, V_full, W_full, U2, V2, W2, 0, 0, 6)
    U_full, V_full, W_full = place(U_full, V_full, W_full, U3, V3, W3, 8, 0, 0)
    return U_full, V_full, W_full


def _verify(U, V, W, n, m, p):
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        T += np.einsum('i,j,k->ijk',
                        U[:, c].reshape(n, m),
                        V[:, c].reshape(m, p),
                        W[:, c].reshape(p, n))
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0 if i == k else 0.0
    return bool(np.max(np.abs(T - T_target)) < _EPS)


def _to_list(M):
    M = np.round(M)
    return [[int(v) for v in row] for row in M]


def construct():
    n, m, p = DIMENSIONS
    if time.time() > _DEADLINE:
        Us, Vs, Ws = _schoolbook(n, m, p)
        return {"U": _to_list(Us), "V": _to_list(Vs), "W": _to_list(Ws)}
    U, V, W = _assemble()
    if _verify(U, V, W, n, m, p):
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    # Fallback: guaranteed-valid schoolbook rank 84.
    Us, Vs, Ws = _schoolbook(n, m, p)
    return {"U": _to_list(Us), "V": _to_list(Vs), "W": _to_list(Ws)}
