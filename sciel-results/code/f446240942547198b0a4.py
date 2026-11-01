"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
import time
import numpy as np


def _schoolbook(n, m, p):
    """Deterministic rank n*m*p decomposition (always valid)."""
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
    return U, V, W


def _als_decompose(n, m, p, rank, seed, max_iter, tol=1e-9, eps=1e-12):
    """Vectorized ALS for the matrix-multiplication tensor.

    Returns (U, V, W) as float64 arrays if the residual drops below tol,
    else None.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)

    # Precompute index maps (row -> (i,j,k))
    Uij = np.array([(a // m, a % m) for a in range(n * m)], dtype=np.int64)  # (nm,2)
    Vjk = np.array([(b // p, b % p) for b in range(m * p)], dtype=np.int64)  # (mp,2)
    Wki = np.array([(c // n, c % n) for c in range(p * n)], dtype=np.int64)  # (pn,2)

    # Target: 1 if (i,j,k) valid (j==k), else 0.  Shape (nm, mp, pn)
    iU = Uij[:, 0]  # (nm,)
    jU = Uij[:, 1]  # (nm,)
    jV = Vjk[:, 0]  # (mp,)
    kV = Vjk[:, 1]  # (mp,)
    kW = Wki[:, 0]  # (pn,)
    iW = Wki[:, 1]  # (pn,)

    # valid[a,b,c] = 1 iff jU[a] == jV[b] == kV[b] and kV[b] == kW[c] and iW[c]==iU[a]
    # jU[a]==jV[b]:  (nm,1) == (1,mp)
    jmatch = (jU[:, None] == jV[None, :]).astype(np.float64)          # (nm,mp)
    kmatch = (kV[:, None] == kW[None, :]).astype(np.float64)          # (mp,pn)
    imatch = (iU[:, None] == iW[None, :]).astype(np.float64)          # (nm,pn)
    target = jmatch * kmatch[None, :, :] * imatch[:, None, :]         # (nm,mp,pn)

    T = target  # alias

    def _residual(U, V, W):
        # r[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
        r = np.einsum('ad,bd,cd->abc', U, V, W)
        return float(np.sum((r - T) ** 2))

    for it in range(max_iter):
        # --- update U: for each a, U[a,:] = num/(den+eps) ---
        # num[a,d] = sum_{b,c} T[a,b,c] V[b,d] W[c,d]
        num = np.einsum('abc,bd,cd->ad', T, V, W)
        den = np.einsum('abc,bd,cd,bd,cd->ad', T, V, W, V, W)
        U = num / (den + eps)

        # --- update V ---
        num = np.einsum('abc,ad,cd->bd', T, U, W)
        den = np.einsum('abc,ad,cd,ad,cd->bd', T, U, W, U, W)
        V = num / (den + eps)

        # --- update W ---
        num = np.einsum('abc,ad,bd->cd', T, U, V)
        den = np.einsum('abc,ad,bd,ad,bd->cd', T, U, V, U, V)
        W = num / (den + eps)

        if it % 20 == 0:
            if _residual(U, V, W) < tol:
                return U, V, W

    if _residual(U, V, W) < tol:
        return U, V, W
    return None


def _round_to_halfint(A):
    """Round each entry to nearest 0.5 (integer/half-integer)."""
    return np.round(A * 2.0) / 2.0


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Exact verification against the tensor (using current float values)."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = np.einsum('ad,bd,cd->abc', U, V, W)
    Uij = np.array([(a // m, a % m) for a in range(n * m)])
    Vjk = np.array([(b // p, b % p) for b in range(m * p)])
    Wki = np.array([(c // n, c % n) for c in range(p * n)])
    iU = Uij[:, 0]; jU = Uij[:, 1]
    jV = Vjk[:, 0]; kV = Vjk[:, 1]
    kW = Wki[:, 0]; iW = Wki[:, 1]
    T = (jU[:, None] == jV[None, :]).astype(np.float64) \
        * (kV[:, None] == kW[None, :]).astype(np.float64)
    T = T * (iU[:, None] == iW[None, :]).astype(np.float64)[:, None, :]
    return float(np.max(np.abs(r - T))) < tol


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # safety margin vs 150s kill

    best = None  # (rank, U, V, W)

    # Deterministic schoolbook as a guaranteed fallback (rank n*m*p = 44).
    U0, V0, W0 = _schoolbook(n, m, p)
    best = (n * m * p, U0, V0, W0)

    # Search for lower ranks using vectorized ALS.
    # Lower bound: tensor has 44 non-zero entries but structure allows lower rank.
    # Try ranks from a low value up to schoolbook-1.
    low = 8
    high = n * m * p - 1
    for rank in range(low, high + 1):
        if time.time() > deadline - 8.0:
            break
        found = False
        for seed in range(0, 200):
            if time.time() > deadline - 8.0:
                break
            res = _als_decompose(n, m, p, rank, seed=seed, max_iter=300)
            if res is not None:
                U, V, W = res
                # Round to half-integers and re-verify exactness.
                Ur, Vr, Wr = _round_to_halfint(U), _round_to_halfint(V), _round_to_halfint(W)
                if _verify(Ur, Vr, Wr, n, m, p):
                    best = (rank, Ur, Vr, Wr)
                    found = True
                    break
                # Also accept unrounded if it verifies (should be ~integer already)
                if _verify(U, V, W, n, m, p):
                    best = (rank, U, V, W)
                    found = True
                    break
        if found:
            # Found a valid decomposition at this rank; no need to try higher
            # ranks for this search direction (we go low->high).
            break

    # Final safety: ensure best is valid; otherwise use schoolbook.
    rank, U, V, W = best
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
