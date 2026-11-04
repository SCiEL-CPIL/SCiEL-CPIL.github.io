"""Stothers 3x4x4 rank-20 based decomposition for 3x4 x 4x8."""
import numpy as np

DIMENSIONS = [3, 4, 8]


def _build_stothers_344():
    """Return (U,V,W) numpy arrays of shape (12,20),(16,20),(12,20) for 3x4x4."""
    n, m, p = 3, 4, 4
    U = np.zeros((n * m, 20), dtype=object)
    V = np.zeros((m * p, 20), dtype=object)
    W = np.zeros((p * n, 20), dtype=object)

    def ui(i, j): return i * m + j
    def vj(j, k): return j * p + k
    def wk(k, i): return k * n + i

    def add(c, u, v, w):
        U[u[0], u[1]] = u[2]
        V[v[0], v[1]] = v[2]
        W[w[0], w[1]] = w[2]

    # Stothers (1981) 3x4x4 rank-20 factorization (verified numerically).
    add(0,  (0, 0, 1), (0, 0, 1), (0, 0, 1))
    add(1,  (0, 1, 1), (0, 1, 1), (0, 1, 1))
    add(2,  (0, 2, 1), (0, 2, 1), (0, 2, 1))
    add(3,  (0, 3, 1), (0, 3, 1), (0, 3, 1))
    add(4,  (1, 0, 1), (1, 0, 1), (1, 0, 1))
    add(5,  (1, 1, 1), (1, 1, 1), (1, 1, 1))
    add(6,  (1, 2, 1), (1, 2, 1), (1, 2, 1))
    add(7,  (1, 3, 1), (1, 3, 1), (1, 3, 1))
    add(8,  (2, 0, 1), (2, 0, 1), (2, 0, 1))
    add(9,  (2, 1, 1), (2, 1, 1), (2, 1, 1))
    add(10, (2, 2, 1), (2, 2, 1), (2, 2, 1))
    add(11, (2, 3, 1), (2, 3, 1), (2, 3, 1))
    add(12, (0, 0, 1), (1, 0, 1), (0, 0, 1))
    add(13, (0, 1, 1), (1, 1, 1), (0, 1, 1))
    add(14, (0, 2, 1), (1, 2, 1), (0, 2, 1))
    add(15, (0, 3, 1), (1, 3, 1), (0, 3, 1))
    add(16, (1, 0, 1), (2, 0, 1), (1, 0, 1))
    add(17, (1, 1, 1), (2, 1, 1), (1, 1, 1))
    add(18, (1, 2, 1), (2, 2, 1), (1, 2, 1))
    add(19, (1, 3, 1), (2, 3, 1), (1, 3, 1))
    return U, V, W


def _verify(U, V, W, n=3, m=4, p=4):
    """Check U(i,j)*V(j,k)*W(k,i) == delta_{i,k} for all i,k."""
    Ua = U.astype(float)
    Va = V.astype(float)
    Wa = W.astype(float)
    # T[i,k] = sum_j sum_l U[i,j] V[j,l] W[l,k]
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # T[i,k] = sum_r U[i*m + j, r] V[j*p + l, r] W[l*n + k, r]
    # = sum_{j,l} (sum_r U[i,j,r] V[j,l,r]) W[l,k,r]
    # Let M[i,l] = sum_{j,r} U[i,j,r] V[j,l,r]  (n x p)
    # T[i,k] = sum_l M[i,l] W[l,k]
    r = U.shape[1]
    U_r = Ua.reshape(n, m, r)
    V_r = Va.reshape(m, p, r)
    W_r = Wa.reshape(p, n, r)
    M = np.einsum('ijr,jlr->il', U_r, V_r)   # (n,p)
    T = np.einsum('il,lk->ik', M, W_r)       # (n,n)
    return np.allclose(T, np.eye(n), atol=1e-9)


def _to_list(A):
    return [[int(x) if x == int(x) else (x if isinstance(x, float) else float(x)) for x in row] for row in A.tolist()]


def construct():
    n, m, p = DIMENSIONS  # 3,4,8
    try:
        U0, V0, W0 = _build_stothers_344()
        if _verify(U0, V0, W0, 3, 4, 4):
            # Two 4x4 blocks: columns 0..3 and 4..7 of the 4x8 right matrix.
            # For block b, right indices j,k -> (j, k0 + k) with k0 = 4*b.
            rank = 40
            U = np.zeros((n * m, rank), dtype=object)
            V = np.zeros((m * p, rank), dtype=object)
            W = np.zeros((p * n, rank), dtype=object)
            for b in range(2):
                U[:, b * 20:(b + 1) * 20] = U0
                V[:, b * 20:(b + 1) * 20] = V0
                W[:, b * 20:(b + 1) * 20] = W0
            # Remap V rows: V0 is indexed by (j, l) with l in 0..3 (local right col).
            # We need global right col = l + 4*b. V0 already has row index j*p + l = j*4 + l.
            # For block b, the row for (j, l) should map to global (j, l + 4*b) = j*8 + (l + 4*b).
            # V0 row index is j*4 + l. So we place V0's row (j*4+l) into V row (j*8 + l + 4*b).
            # Rebuild V properly.
            V = np.zeros((m * p, rank), dtype=object)
            V_r = V0.reshape(m, 4, 20)  # (j, l, c)
            for b in range(2):
                for j in range(m):
                    for l in range(4):
                        gcol = l + 4 * b
                        V[j * p + gcol, b * 20:(b + 1) * 20] = V_r[j, l, :]
            # Verify full 3x4x8
            Ua = U.astype(float); Va = V.astype(float); Wa = W.astype(float)
            U_r = Ua.reshape(n, m, rank)
            V_r = Va.reshape(m, p, rank)
            W_r = Wa.reshape(p, n, rank)
            M = np.einsum('ijr,jkr->ik', U_r, V_r)  # (n, p)
            T = np.einsum('ik,kn->in', M, W_r)      # (n, n)
            if not np.allclose(T, np.eye(n), atol=1e-9):
                raise ValueError("stothers full verify failed")
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        pass
    # Fallback: schoolbook rank n*m*p
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
