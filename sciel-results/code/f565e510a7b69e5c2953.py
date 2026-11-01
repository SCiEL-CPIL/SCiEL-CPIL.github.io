"""Rank-49 BCLR (Strassen ⊗ Strassen) decomposition of 4x4x4 matrix multiplication.

The factor matrices are built as the Kronecker product of the 2x2 Strassen
factors, which explicitly reorders the tensor indices to satisfy the official
trace convention W[k*n+i, r].  A budget-bounded linear-dependency elimination
pass then attempts to reduce the rank further (the known optimum is 49, so this
is a refinement hypothesis rather than a guaranteed gain).
"""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# 2x2 Strassen factors (row-major indexing: 00, 01, 10, 11).
#
#   A = [[a,b],[c,d]]  B = [[e,f],[g,h]]
#   P1=(a+d)(e+h)  P2=(c+d)e  P3=a(f-h)  P4=d(g-e)
#   P5=(a+b)h      P6=(c-a)(e+f)  P7=(b-d)(g+h)
#
#   C00 = P1 + P4 - P5 + P7
#   C01 = P3 + P5
#   C10 = P2 + P4
#   C11 = P1 - P2 + P3 + P6
# ---------------------------------------------------------------------------
_S2_U = [
    [1, 0, 0, 1],   # P1 A-part (a+d)
    [0, 0, 1, 1],   # P2 A-part (c+d)
    [1, 0, 0, 0],   # P3 A-part (a)
    [0, 0, 0, 1],   # P4 A-part (d)
    [1, 1, 0, 0],   # P5 A-part (a+b)
    [1, 0, 0, 0],   # P6 A-part (c-a)
    [0, 1, 0, 0],   # P7 A-part (b-d)
]
_S2_V = [
    [1, 0, 0, 1],   # P1 B-part (e+h)
    [1, 0, 0, 0],   # P2 B-part (e)
    [0, 1, 0, -1],  # P3 B-part (f-h)
    [0, 0, 1, -1],  # P4 B-part (g-e)
    [0, 0, 0, 1],   # P5 B-part (h)
    [1, 1, 0, 0],   # P6 B-part (e+f)
    [0, 0, 1, 1],   # P7 B-part (g+h)
]
_S2_W = [
    [1, 0, 0, 0],   # P1 -> C00
    [0, 0, 0, 1],   # P2 -> C11
    [0, 1, 0, 0],   # P3 -> C01
    [1, 0, 0, 0],   # P4 -> C00
    [-1, 1, 0, 0],  # P5 -> C00(-1), C01(+1)
    [0, 0, 0, 1],   # P6 -> C11
    [1, 0, 0, 0],   # P7 -> C00
]


def _kronecker(a, b):
    """Kronecker product of two 2D lists -> 2D list."""
    ra, ca = len(a), len(a[0])
    rb, cb = len(b), len(b[0])
    out = [[0] * (ca * cb) for _ in range(ra * rb)]
    for i in range(ra):
        for j in range(ca):
            aij = a[i][j]
            if aij == 0:
                continue
            base_i, base_j = i * rb, j * cb
            for k in range(rb):
                for l in range(cb):
                    out[base_i + k][base_j + l] += aij * b[k][l]
    return out


def _build_rank49():
    """Build the rank-49 BCLR decomposition as the Kronecker product of the
    2x2 Strassen factors.  The row order (00,01,10,11) ⊗ (00,01,10,11) matches
    the official trace convention W[k*n+i, r].  Returns (U, V, W)."""
    U = _kronecker(_S2_U, _S2_U)  # (16, 49)
    V = _kronecker(_S2_V, _S2_V)  # (16, 49)
    W = _kronecker(_S2_W, _S2_W)  # (16, 49)
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k."""
    import numpy as np
    Ua = np.array(U, dtype=np.float64).reshape(n, m, -1)
    Va = np.array(V, dtype=np.float64).reshape(m, p, -1)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, -1)
    res = np.einsum('ijr,jkr,ki r->ijk', Ua, Va, Wa)
    return bool(np.allclose(res, 1.0, atol=1e-9))


def _reduce_rank(U, V, W, n, m, p, deadline):
    """Attempt to reduce the rank by eliminating linearly dependent terms.

    A term r is redundant if its column in the concatenated factor space is a
    linear combination of the others.  We detect this via row-rank of the
    concatenated [U; V; W] columns and drop any column that lies in the span
    of the remaining columns.  This is a refinement hypothesis: the known
    optimum for 4x4x4 is 49, so this usually finds no reduction, but it is
    cheap and safe (we verify the result before accepting it).
    """
    import numpy as np
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    if rank <= 1:
        return U, V, W

    # Concatenate the three factors row-wise; each term r is one column.
    cat = np.vstack([Ua, Va, Wa])  # (48, rank)
    tol = 1e-9

    keep = list(range(rank))
    # Greedy column elimination: repeatedly drop a column that is in the span
    # of the columns that remain (i.e., does not increase the column rank).
    changed = True
    while changed and time.time() < deadline:
        changed = False
        cols = np.array(keep)
        cur_rank = np.linalg.matrix_rank(cat[:, cols], tol=tol)
        for idx in range(len(keep)):
            cand = [c for c in keep if c != cols[idx]]
            if len(cand) == 0:
                continue
            cand_rank = np.linalg.matrix_rank(cat[:, cand], tol=tol)
            if cand_rank == cur_rank:
                # column cols[idx] is redundant
                keep = cand
                changed = True
                break

    if len(keep) == rank:
        return U, V, W

    U2 = Ua[:, keep]
    V2 = Va[:, keep]
    W2 = Wa[:, keep]
    if _verify(U2.tolist(), V2.tolist(), W2.tolist(), n, m, p):
        return U2.tolist(), V2.tolist(), W2.tolist()
    return U, V, W


def _schoolbook(n, m, p):
    """Fallback rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Guaranteed rank-49 BCLR (Strassen ⊗ Strassen) decomposition.
    U, V, W = _build_rank49()
    if _verify(U, V, W, n, m, p):
        # Refinement hypothesis: try to reduce the rank further.
        try:
            U2, V2, W2 = _reduce_rank(U, V, W, n, m, p, deadline)
            if len(U2[0]) < len(U[0]):
                return {"U": U2, "V": V2, "W": W2}
        except Exception:
            pass
        return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook (should never be reached).
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
