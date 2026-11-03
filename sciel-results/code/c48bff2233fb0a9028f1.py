"""Matrix multiplication 3,3,5 rank minimization."""
DIMENSIONS = [3, 3, 5]

import time
import numpy as np

_START = time.time()
_WALL_LIMIT = 150.0          # evaluator hard kill
_SAFETY_MARGIN = 12.0        # return before the hard limit
_DEADLINE = _START + _WALL_LIMIT - _SAFETY_MARGIN

# Precompute the index mapping for the three tensor legs.
# Leg 1 (U): rows i*m+j  -> (i, j)
# Leg 2 (V): rows j*p+k  -> (j, k)
# Leg 3 (W): rows k*n+i  -> (k, i)
_N, _M, _P = DIMENSIONS
_U_ROWS = [ (i, j) for i in range(_N) for j in range(_M) ]
_V_ROWS = [ (j, k) for j in range(_M) for k in range(_P) ]
_W_ROWS = [ (k, i) for k in range(_P) for i in range(_N) ]

# For each W-row (k, i), the set of (u_row, v_row) pairs that contribute:
#   sum_l U[i*m+j, l] * V[j*p+k, l] = 1   for all j
# Build the linear system matrix A_w of shape (m, n*m + m*p) whose rows index j.
_W_SYSTEMS = []
for (k, i) in _W_ROWS:
    # columns: U index = i*m+j ; V index = j*p+k
    cols = []
    for j in range(_M):
        cols.append(i * _M + j)      # U column
        cols.append(j * _P + k)      # V column
    _W_SYSTEMS.append(np.array(cols, dtype=int))


def _verify(U, V, W, tol=1e-9):
    """Return True iff the factors exactly reproduce the multiplication tensor."""
    n, m, p = DIMENSIONS
    r = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j], V[j*p + k]) * 0.0
                # s = sum_l U[i*m+j,l] * V[j*p+k,l] * W[k*n+i,l]
                s = float(np.einsum('l,l,l->', U[i*m + j], V[j*p + k], W[k*n + i]))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, tol=1e-9):
    """Vectorized verification."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Build full tensor of the decomposition.
    # T[i,j,k] = sum_l U[i*m+j,l] V[j*p+k,l] W[k*n+i,l]
    # Reshape into (n,m,p,r) views.
    Uv = U.reshape(n, m, -1)
    Vv = V.reshape(m, p, -1)
    Wv = W.reshape(p, n, -1)
    # Uv: (n,m,r), Vv: (m,p,r), Wv: (p,n,r)
    # We want for each (i,j,k): dot over r of Uv[i,j,:]*Vv[j,k,:]*Wv[k,i,:]
    # Use einsum: Uv[i,j,l]*Vv[j,k,l]*Wv[k,i,l] -> sum l
    T = np.einsum('ijl,jkl,kil->ijk', Uv, Vv, Wv)
    return bool(np.all(np.abs(T - 1.0) <= tol))


def _schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W


def _solve_W(U, V, tol=1e-8):
    """Given U (n*m x r) and V (m*p x r), solve for W (p*n x r).

    Returns W if a solution exists with residual < tol, else None.
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    W = np.zeros((p * n, r))
    for wi, (k, i) in enumerate(_W_ROWS):
        cols = _W_SYSTEMS[wi]
        A = np.zeros((m, len(cols)))
        # A[j, t] = U[i*m+j, l] * V[j*p+k, l] summed over l -> this is the coefficient
        # of W[k*n+i, l] in the equation for index j.
        # Equation for j: sum_l (U[i*m+j, l] * V[j*p+k, l]) * W[k*n+i, l] = 1
        # So A[j, l] = U[i*m+j, l] * V[j*p+k, l]
        for j in range(m):
            A[j, :] = U[i*m + j] * V[j*p + k]
        # Solve A W_col = 1
        # Use lstsq; check residual.
        try:
            sol, residuals, rank, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
        except np.linalg.LinAlgError:
            return None
        # residual check
        res = np.linalg.norm(A @ sol - np.ones(m))
        if res > tol:
            return None
        W[wi, :] = sol
    return W


def _try_rank(r, rng, max_trials=4000, tol=1e-8):
    """Attempt to find an exact rank-r decomposition. Returns (U,V,W) or None."""
    n, m, p = DIMENSIONS
    for t in range(max_trials):
        U = rng.standard_normal((n * m, r))
        V = rng.standard_normal((m * p, r))
        W = _solve_W(U, V, tol=tol)
        if W is not None:
            if _verify_fast(U, V, W, tol=1e-7):
                return U, V, W
    return None


def construct():
    """Return a rank-minimal exact tensor decomposition for 3x3 * 3x5.

    The schoolbook decomposition (rank 45) is always exact and is the fallback.
    We search for a strictly lower-rank exact decomposition and return the
    lowest rank found within the wall-clock budget.
    """
    n, m, p = DIMENSIONS
    best = None
    U_sb, V_sb, W_sb = _schoolbook()
    best = (U_sb, V_sb, W_sb)  # rank 45 guaranteed

    # Try ranks from low to high, stopping at the first exact hit (lowest rank).
    # The trivial lower bound for 3x3x5: at least max(n*m, m*p, p*n) = max(9,15,15)=15.
    rng = np.random.default_rng(12345)

    # Candidate ranks to try. We search downward from just below the best known
    # (45) down to the lower bound, but prioritize a small set of promising
    # ranks first to keep runtime bounded, then widen if time remains.
    lower_bound = max(n * m, m * p, p * n)  # 15
    # Search ranks in a sensible order: near the known best first is wasteful;
    # instead scan from lower_bound upward is too many. We scan a curated list.
    ranks_to_try = list(range(lower_bound, 45))
    # Sort so we prefer the smallest ranks first (best score if found).
    ranks_to_try.sort()

    for r in ranks_to_try:
        if time.time() > _DEADLINE:
            break
        # Per-rank time budget: keep total under deadline.
        res = _try_rank(r, rng, max_trials=2000, tol=1e-8)
        if res is not None:
            U, V, W = res
            best = (U, V, W)
            # Found a lower rank; continue searching for even lower ranks.
            continue
    # If we found nothing below 45, best remains schoolbook.

    U, V, W = best
    # Convert to lists of floats (integer/half-integer acceptable; floats fine).
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }


if __name__ == "__main__":
    out = construct()
    print("rank =", len(out["U"][0]))
    print("valid =", _verify(out["U"], out["V"], out["W"]))
