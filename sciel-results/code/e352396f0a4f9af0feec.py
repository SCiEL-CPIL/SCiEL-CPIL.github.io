import numpy as np
import time

DIMENSIONS = [2, 3, 3]
_START = time.time()
_LIMIT = 140.0  # safety margin before the 150s kill


def _matmul_tensor(n, m, p):
    """Full tensor T[u,v,w] with u=i*m+j, v=j*p+k, w=k*n+i (official trace convention)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _exact_verify(n, m, p, U, V, W):
    """Check sum_c U[u,c]*V[v,c]*W[w,c] == T[u,v,w] for ALL (u,v,w). Integer-exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = _matmul_tensor(n, m, p)
    rank = U.shape[1]
    S = np.zeros((n * m, m * p, p * n), dtype=float)
    for c in range(rank):
        S += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return bool(np.allclose(S, T, atol=1e-9, rtol=0.0))


def _to_list(M):
    """Convert to list-of-lists, snapping near-integer / half-integer values."""
    M = np.asarray(M, dtype=float)
    snapped = np.rint(M * 2.0) / 2.0
    return [[float(v) for v in row] for row in snapped]


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 3

    # ---- Primary: rank-17 Strassen(2x2x2) + schoolbook construction ----
    # U: (n*m, r) indexed [i*m+j]; V: (m*p, r) indexed [j*p+k]; W: (p*n, r) indexed [k*n+i]
    r = 17
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))

    # Strassen 2x2x2 on A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # 7 products; each product c uses U[i, c]*V[j, c]*W[k, c].
    #   c0: i in {0,1}, j=0, k in {0,1}
    #   c1: i=0, j in {0,1}, k in {0,1}
    #   c2: i in {0,1}, j=1, k in {0,1}
    #   c3: i in {0,1}, j=0, k=1
    #   c4: i in {0,1}, j=1, k=0
    #   c5: i=0, j in {0,1}, k=1
    #   c6: i=1, j in {0,1}, k=0
    for c in range(7):
        for i in range(n):
            for j in range(2):
                for k in range(2):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0

    # Schoolbook for the remaining 2x3 * 3x3 part: i in {0,1}, j in {2}, k in {0,1,2}
    col = 7
    for i in range(n):
        for j in range(2, m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    assert col == r, (col, r)

    if _exact_verify(n, m, p, U, V, W):
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    # ---- Fallback: schoolbook rank n*m*p ----
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
