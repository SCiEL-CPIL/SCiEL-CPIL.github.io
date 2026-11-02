"""Search for a rank-minimal exact tensor decomposition of 2x3 * 3x6 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin vs 150s kill


def _time_ok():
    return time.time() < _DEADLINE


def construct():
    """Return {'U','V','W'} with integer/half-integer entries exactly decomposing the tensor."""
    n, m, p = DIMENSIONS
    best = _schoolbook()
    if not _time_ok():
        return best

    # Search for lower-rank exact decompositions via randomized rank reduction.
    rng = np.random.default_rng(12345)
    for r in range(35, 30, -1):
        if not _time_ok():
            break
        found = _try_rank(r, rng)
        if found is not None:
            best = found
            break  # try to go lower
        # a few retries at this rank before dropping
        for _ in range(2):
            if not _time_ok():
                break
            found = _try_rank(r, rng)
            if found is not None:
                best = found
                break
        if best is not found and best is not None and _rank_of(best) < 36:
            break
    if not _time_ok():
        return best
    # polish: try to reduce further from best
    cur = _rank_of(best)
    for r in range(cur - 1, max(cur - 4, 1), -1):
        if not _time_ok():
            break
        found = _try_rank(r, rng)
        if found is not None:
            best = found
            break
    return best


def _rank_of(result):
    if result is None:
        return 36
    return len(result["U"][0])


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _try_rank(r, rng):
    """Attempt to find an exact decomposition of rank r. Returns dict or None."""
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n

    # Build target tensor T (N*M*P) where T[i,j,k]=1 iff k*n+i == j*m+k (i.e. i==j and k==k).
    # Actually condition: output index (i,k) = i*m + j and input (j,k) => j == i.
    # So T[i, j, k] = 1 if i == j else 0.
    T = np.zeros((N, M, P), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            if i == j:
                for k in range(P):
                    T[i, j, k] = 1.0

    for attempt in range(3):
        if not _time_ok():
            return None
        # Random U (N x r) and W (P x r)
        U = rng.standard_normal((N, r))
        W = rng.standard_normal((P, r))
        # For each k, solve U^T V_k W^T = T[:,:,k] for V_k (M x r)
        # Reshape: (N, M) = U (N,r) * V_k (r,M) * W^T (r,P) -> (N,P)?? 
        # Careful: T[i,j,k] with i in N, j in M, k in P.
        # We want sum_c U[i,c] V[j,c] W[k,c] = T[i,j,k].
        # For fixed k: matrix A(i,j) = T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c].
        # Let w_k = W[:,k] (r,). Then A = U * diag(w_k) * V^T? Check: (U diag(w) V^T)[i,j] = sum_c U[i,c] w_c V[j,c]. Yes!
        # So A_k = U * diag(w_k) * V_k^T, where V_k = V[:,k] (M,).
        # Solve for V_k: A_k = U * diag(w_k) * V_k^T.
        # A_k = (U * diag(w_k)) * V_k^T.
        # Let B = U * diag(w_k) (N x r). Then A_k = B * V_k^T => A_k^T = V_k * B^T.
        # Solve V_k (M x r) from A_k^T (M x N) = V_k (M x r) B^T (r x N).
        # This is a linear system: for each i, A_k[i,:] = V_k[i,:] B^T.
        # Stack: A_k^T (M,N) = V_k (M,r) * B^T (r,N).
        # Solve via least squares on the Kronecker: V_k = A_k^T @ pinv(B^T).
        # pinv(B^T) is (N x r). A_k^T (M x N) @ (N x r) = (M x r).
        for k in range(P):
            w_k = W[:, k]
            B = U * w_k  # (N x r)
            A_k = T[:, :, k]  # (N x M)
            # A_k (N,M) = B (N,r) * V_k^T (r,M) => V_k^T = pinv(B) @ A_k => V_k = A_k^T @ pinv(B)^T
            # Use least squares: minimize ||B * X - A_k|| where X = V_k^T (r x M).
            # B (N x r), X (r x M): solve B X = A_k.
            try:
                X, *_ = np.linalg.lstsq(B, A_k, rcond=None)  # X (r x M)
            except np.linalg.LinAlgError:
                X = None
            if X is None:
                continue
            V_k = X.T  # (M x r)
            # store
            if attempt == 0:
                V = np.zeros((M, r))
            V[:, k] = V_k
        # Now we have U, V, W. Verify exactness over rationals.
        # Build candidate with rational reconstruction from floats.
        cand = _to_rational_candidate(U, V, W)
        if cand is None:
            continue
        if _verify(cand):
            return cand
    return None


def _to_rational_candidate(U, V, W):
    """Reconstruct U,V,W entries as Fractions from floats, return dict or None."""
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    r = U.shape[1]
    def recon(arr):
        out = []
        for row in arr:
            rrow = []
            for val in row:
                fr = Fraction(val).limit_denominator(10**6)
                rrow.append(fr)
            out.append(rrow)
        return out
    return {"U": recon(U), "V": recon(V), "W": recon(W)}


def _verify(cand):
    """Check exact decomposition over rationals."""
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    r = len(cand["U"][0])
    U, V, W = cand["U"], cand["V"], cand["W"]
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = Fraction(0)
                for c in range(r):
                    s += U[i][c] * V[j][c] * W[k][c]
                expected = 1 if i == j else 0
                if s != expected:
                    return False
    return True


if __name__ == "__main__":
    construct()
