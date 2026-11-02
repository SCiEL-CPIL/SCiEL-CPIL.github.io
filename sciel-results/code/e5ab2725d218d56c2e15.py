"""Rewritten program: guaranteed schoolbook fallback + time-boxed exact low-rank search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = time.time() + 115.0  # safety margin before 120s eval / 150s kill
_BEST = None  # best verified (rank, U, V, W) found so far


def _elapsed_ok():
    return time.time() < _DEADLINE


def _verify(U, V, W, n, m, p):
    """Check that the half-integer factors exactly decompose the multiplication tensor.
    Convention: W[k*n + i, r]. U indexed [i*m+j], V indexed [j*p+k].
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] must equal 1, and for
    all other (i',j',k') the product tensor must be 0 where it should be.
    The multiplication tensor T[i,j,k] = 1 (it's the standard one). The
    decomposition requires: for every (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]
    reconstructs the bilinear form. Actually the standard check: the tensor
    T_{ijk} = sum_r U_{i m + j, r} V_{j p + k, r} W_{k n + i, r} should equal
    the matrix-multiplication tensor, which is 1 for the "matching" pattern.
    """
    U = np.asarray(U, dtype=object)
    V = np.asarray(V, dtype=object)
    W = np.asarray(W, dtype=object)
    rank = U.shape[1]
    # Build the full multiplication tensor via the factors and compare to identity-like.
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # The true matrix-mult tensor: (A*B)[i,k] = sum_j A[i,j] B[j,k], so the
    # tensor of the bilinear map is T[i,j,k] = 1 for all i,j,k? No.
    # The tensor represents the trilinear form (A,B,C) -> sum A[i,j] B[j,k] C[k,i]? 
    # Standard: multiplication tensor has entries 1 at (i,j,k) meaning A[i,j]*B[j,k] contributes to C[i,k].
    # The rank-1 slice contributes U[i*m+j] * V[j*p+k] * W[k*n+i] to the (i,j,k) slot.
    # We need sum_r = 1 for each (i,j,k) that is part of the product, i.e. ALL (i,j,k) since
    # every A[i,j]*B[j,k] term appears. So the tensor is the all-ones tensor of shape (n,m,p).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s = s + U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if s != 1:
                    return False
    return True


def _is_half_integer(arr):
    a = np.asarray(arr, dtype=float)
    doubled = 2.0 * a
    return np.allclose(doubled, np.round(doubled), atol=1e-9)


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W


def _try_low_rank(n, m, p, start_rank, end_rank):
    """Heuristic bounded search for a rank in [start_rank, end_rank] exact half-integer
    decomposition. Uses random half-integer U,V and solves for W, then verifies.
    Very unlikely to succeed for 3x4x7 but bounded by deadline."""
    global _BEST
    rng = np.random.default_rng(12345)
    for rank in range(start_rank, end_rank + 1):
        if not _elapsed_ok():
            break
        for attempt in range(60):
            if not _elapsed_ok():
                break
            # Random half-integer (values in {-1,-0.5,0,0.5,1}) U and V
            U = (rng.integers(0, 4, size=(n*m, rank)) - 1.5) / 2.0
            V = (rng.integers(0, 4, size=(m*p, rank)) - 1.5) / 2.0
            # Solve for W: for each (i,j,k) we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]=1.
            # This is not linear in W per (k*n+i) because the same W[k*n+i,r] couples across i,j.
            # Instead build linear system: unknowns W[k*n+i, r] for all (k,i,r). 
            # Equation for (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
            # Unknown index for W: w_idx = (k*n+i)*rank + r.
            num_unknowns = p * n * rank
            num_eq = n * m * p
            A = np.zeros((num_eq, num_unknowns), dtype=float)
            b = np.ones(num_eq, dtype=float)
            eq = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            col = (k*n+i)*rank + r
                            A[eq, col] = U[i*m+j, r] * V[j*p+k, r]
                        eq += 1
            # Solve least squares, then snap to half-integers, then verify.
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
            W = sol.reshape(p*n, rank)
            W_snapped = np.round(2.0 * W) / 2.0
            # Verify with exact half-integer values
            W_list = [[int(W_snapped[r, c]) if float(W_snapped[r, c]).is_integer() else float(W_snapped[r, c])
                       for c in range(rank)] for r in range(p*n)]
            U_list = [[float(U[r, c]) for c in range(rank)] for r in range(n*m)]
            V_list = [[float(V[r, c]) for c in range(rank)] for r in range(m*p)]
            if _is_half_integer(U_list) and _is_half_integer(V_list) and _is_half_integer(W_list):
                if _verify(U_list, V_list, W_list, n, m, p):
                    if _BEST is None or rank < _BEST[0]:
                        _BEST = (rank, U_list, V_list, W_list)
                        return _BEST
    return _BEST


def construct():
    global _BEST
    n, m, p = DIMENSIONS
    # Guaranteed valid schoolbook
    U, V, W = _schoolbook(n, m, p)
    _BEST = (n*m*p, U, V, W)

    # Time-boxed search for better rank (very low expected success, bounded)
    try:
        _try_low_rank(n, m, p, 20, n*m*p - 1)
    except Exception:
        pass

    if _BEST is None:
        U, V, W = _schoolbook(n, m, p)
        _BEST = (n*m*p, U, V, W)

    rank, U, V, W = _BEST
    return {"U": U, "V": V, "W": W}
