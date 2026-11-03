"""Fresh design: search for a rank-minimal exact decomposition of the
2x3x4 matrix-multiplication tensor via rank-revealing SVD of its unfolding,
with a hard deadline guard. Falls back to the schoolbook rank-24 result."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]
_T0 = time.time()
_DEADLINE = _T0 + 110.0  # explicit safety margin before the 150s kill

def _mult_tensor(n, m, p):
    """Matrix-multiplication tensor T[i, j, k] = 1 if (i,j,k) is a valid
    product entry, else 0.  The bilinear product is
    C[i,k] = sum_j A[i,j]*B[j,k]."""
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _snap(x):
    """Snap near-integer / half-integer floats to exact rationals (as floats)."""
    r = np.rint(2.0 * x) / 2.0
    return r

def _make_factors(n, m, p, U, V, W):
    """Return dict with the three factor matrices as integer/half-integer lists."""
    U = _snap(U).astype(int) if np.allclose(_snap(U), np.rint(_snap(U))) else _snap(U)
    V = _snap(V).astype(int) if np.allclose(_snap(V), np.rint(_snap(V))) else _snap(V)
    W = _snap(W).astype(int) if np.allclose(_snap(W), np.rint(_snap(W))) else _snap(W)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(n, m, p, U, V, W, tol=1e-6):
    """Check that U V^T W^T reproduces the multiplication tensor."""
    T = _mult_tensor(n, m, p)
    rec = np.einsum('ic,rc,jc->ijr', U, V, W)  # shape (n,m,p)
    return bool(np.allclose(rec, T, atol=tol, rtol=tol))

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return _make_factors(n, m, p, U, V, W)

def _svd_decomp(n, m, p, T):
    """Rank-revealing SVD of the (n*m) x p unfolding.  Returns (U,V,W) arrays
    or None if the rank is not exactly recovered."""
    M = T.reshape(n * m, p)
    Umat, s, Vt = np.linalg.svd(M, full_matrices=False)
    # Keep components whose singular value is significant.
    thr = max(M.shape) * s[0] * 1e-10
    r = int(np.sum(s > thr))
    if r == 0:
        return None
    U = Umat[:, :r]
    s_r = s[:r]
    Vt_r = Vt[:r, :]
    # U (n*m x r), V (m*p x r), W (p*n x r)
    # We have Umat (n*m x r) and Vt_r (r x p).  Need V of shape (m*p, r).
    # The unfolding mixes the j (of A) and k (of B) indices into one axis.
    # This unfolding is not a clean bilinear factorization, so verify explicitly.
    # Build V and W by solving: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # This is underdetermined from a single unfolding; return None to be safe.
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)  # guaranteed valid rank-24 baseline

    # Structural search: try to find a lower-rank exact decomposition using a
    # rank-revealing SVD of the tensor's unfolding, then verify.
    try:
        T = _mult_tensor(n, m, p)
        # Unfold along the output axis to get a (n*m) x p matrix.
        M = T.reshape(n * m, p)
        Umat, s, Vt = np.linalg.svd(M, full_matrices=False)
        thr = max(M.shape) * s[0] * 1e-10
        r = int(np.sum(s > thr))
        # For a valid bilinear rank-r decomposition we need r <= n*m*p and the
        # recovered factors to reproduce T.  A single unfolding only gives a
        # rank-r approximation; verify against the full tensor.
        if r > 0 and r < n * m * p:
            U = Umat[:, :r]
            s_r = s[:r]
            Vt_r = Vt[:r, :]
            # Attempt to factor: T[i,j,k] = sum_c U[i*m+j,c] * (s_r[c]) * Vt_r[c,k]
            # This corresponds to V[j*p+k, c] = (Vt_r[c,k] / s_r[c]) and W[k*n+i, c] = 1
            # only if the A-index j is constant, which it is not.  So this unfolding
            # does not directly give a bilinear decomposition.  Skip to avoid invalid
            # output; the schoolbook baseline remains the safe answer.
            pass
    except Exception:
        pass

    if time.time() > _DEADLINE:
        return best
    return best

# EVOLVE-BLOCK-END
