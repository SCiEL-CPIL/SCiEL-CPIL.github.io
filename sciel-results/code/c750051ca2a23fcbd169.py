"""Rank-minimal exact tensor decomposition for 2x3 times 3x7 matrix multiplication.

The multiplication tensor T[u,v,w] equals 1 iff j_u == j_v and k_v == k_w and
i_u == i_w (with u = i*m+j, v = j*p+k, w = k*n+i), else 0.  This is the product
of three "match" factors.  We search for a rank-r decomposition T = sum_r U_r
(o) V_r (o) W_r with small integer / half-integer entries, using a structured
random search guided by that factorisation, plus an exact rational verification.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]
N, M, P = DIMENSIONS
ROWS_U = N * M          # 6
ROWS_V = M * P          # 21
ROWS_W = P * N          # 14

# Wall-clock safety margin (evaluator kills at ~120-150 s).
DEADLINE = 118.0


def _seed():
    """Schoolbook rank-42 decomposition (fallback)."""
    n, m, p = DIMENSIONS
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


def _check_decomposition(U, V, W, n, m, p):
    """Exact check using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                s = Fraction(0)
                Ur = U[u]; Vr = V[v]; Wr = W[w]
                for r in range(rank):
                    s += Ur[r] * Vr[r] * Wr[r]
                if s != 1:
                    return False
    # Inconsistent triples: T[u,v,w]=0 must hold.
    for u in range(n * m):
        i, j = divmod(u, m)
        for v in range(m * p):
            j2, k = divmod(v, p)
            for w in range(p * n):
                k2, i2 = divmod(w, n)
                if j == j2 and k == k2 and i == i2:
                    continue
                s = Fraction(0)
                Ur = U[u]; Vr = V[v]; Wr = W[w]
                for r in range(rank):
                    s += Ur[r] * Vr[r] * Wr[r]
                if s != 0:
                    return False
    return True


def _fast_check_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float check for candidate pruning."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                s = np.dot(U[u], V[v] * W[w])
                if abs(s - 1.0) > tol:
                    return False
    # Sample inconsistent triples.
    rng = np.random.default_rng(12345)
    for _ in range(200):
        u = rng.integers(n * m)
        v = rng.integers(m * p)
        w = rng.integers(p * n)
        i, j = divmod(u, m)
        j2, k = divmod(v, p)
        k2, i2 = divmod(w, n)
        if j == j2 and k == k2 and i == i2:
            continue
        s = np.dot(U[u], V[v] * W[w])
        if abs(s) > tol:
            return False
    return True


def _structured_search(rank, time_limit, seed):
    """Structured random search exploiting the product-of-deltas factorisation.

    Each rank-1 component is built from indicator vectors over the three index
    sets {i}, {j}, {k}.  A component with "signature" (i_r, j_r, k_r) and
    coefficient c_r contributes c_r * 1_{i_u=i_r} 1_{j_u=j_r} 1_{j_v=j_r}
    1_{k_v=k_r} 1_{k_w=k_r} 1_{i_w=i_r}.  A sum over all 42 signatures with
    coefficient 1 gives the schoolbook.  We allow coefficients to vary and
    allow a component to be a (small) linear combination of signatures, which
    can cancel inconsistent contributions and reduce the effective rank.
    """
    n, m, p = DIMENSIONS
    end_time = time.time() + time_limit
    rng = np.random.default_rng(seed)

    # Precompute, for each row index, the (i, j) / (j, k) / (k, i) labels.
    u_lab = [(i, j) for i in range(n) for j in range(m)]
    v_lab = [(j, k) for j in range(m) for k in range(p)]
    w_lab = [(k, i) for k in range(p) for i in range(n)]

    # All 42 signatures (i, j, k).
    sigs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_sig = len(sigs)

    for attempt in range(200000):
        if time.time() > end_time:
            break

        U = np.zeros((ROWS_U, rank))
        V = np.zeros((ROWS_V, rank))
        W = np.zeros((ROWS_W, rank))

        for r in range(rank):
            # Pick a small number of signatures to combine in this component.
            ncomp = int(rng.integers(1, 5))  # 1..4
            chosen = rng.choice(n_sig, size=ncomp, replace=False)
            for c in chosen:
                i, j, k = sigs[c]
                coeff = float(rng.choice([-1.0, 1.0]))
                # U: nonzero on rows (i, j)
                U[i * m + j, r] += coeff
                # V: nonzero on rows (j, k)
                V[j * p + k, r] += coeff
                # W: nonzero on rows (k, i)
                W[k * n + i, r] += coeff

        if not _fast_check_float(U, V, W, n, m, p):
            continue

        # Exact rational verification.
        Uf = [[Fraction(int(round(x))) for x in row] for row in U]
        Vf = [[Fraction(int(round(x))) for x in row] for row in V]
        Wf = [[Fraction(int(round(x))) for x in row] for row in W]
        if _check_decomposition(Uf, Vf, Wf, n, m, p):
            return {"U": Uf, "V": Vf, "W": Wf}

    return None


def _general_search(rank, time_limit, seed):
    """Broader search: each vector is a sparse vector with small integer entries
    (not restricted to the signature structure), to allow cancellations that the
    pure-signature family cannot express."""
    n, m, p = DIMENSIONS
    end_time = time.time() + time_limit
    rng = np.random.default_rng(seed + 1000003)

    for attempt in range(200000):
        if time.time() > end_time:
            break

        U = np.zeros((ROWS_U, rank))
        V = np.zeros((ROWS_V, rank))
        W = np.zeros((ROWS_W, rank))

        for r in range(rank):
            for mat, rows in ((U, ROWS_U), (V, ROWS_V), (W, ROWS_W)):
                nnz = int(rng.integers(1, 4))
                idx = rng.choice(rows, size=nnz, replace=False)
                for idx_i in idx:
                    mat[idx_i, r] = float(rng.choice([-1.0, 1.0]))

        if not _fast_check_float(U, V, W, n, m, p):
            continue

        Uf = [[Fraction(int(round(x))) for x in row] for row in U]
        Vf = [[Fraction(int(round(x))) for x in row] for row in V]
        Wf = [[Fraction(int(round(x))) for x in row] for row in W]
        if _check_decomposition(Uf, Vf, Wf, n, m, p):
            return {"U": Uf, "V": Vf, "W": Wf}

    return None


def construct():
    start = time.time()

    # Try to beat the schoolbook rank 42.  Search a few candidate ranks.
    for rank in range(38, 42):
        if time.time() > DEADLINE - 12:
            break
        budget = min(12.0, DEADLINE - 12 - (time.time() - start))
        if budget <= 0:
            break
        res = _structured_search(rank, budget, seed=rank)
        if res is not None:
            return res
        if time.time() > DEADLINE - 12:
            break
        budget = min(8.0, DEADLINE - 12 - (time.time() - start))
        if budget <= 0:
            break
        res = _general_search(rank, budget, seed=rank)
        if res is not None:
            return res

    # Fallback: schoolbook rank-42.
    return _seed()
