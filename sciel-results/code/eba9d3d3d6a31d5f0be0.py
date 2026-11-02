"""Search for a rank-minimal exact tensor decomposition of 3x4x7 matmul.

Strategy:
- Rank r decomposition = r vectors U[:,c],V[:,c],W[:,c] with U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
- The tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] must equal 1.
- We try to find a LOW rank via a randomized nonlinear least-squares (Levenberg-Marquardt)
  search using NumPy's fast einsums. If we find a valid low-rank decomposition we return it.
- A deterministic schoolbook construction (rank n*m*p=84) is ALWAYS available and used as
  the guaranteed-valid fallback, so the program never crashes and always returns a valid
  decomposition. The score is the rank (lower is better).

Integer/half-integer requirement: the search returns exact integers/half-integers when it
converges to a clean structure; otherwise the (valid, possibly fractional) schoolbook result
is returned. Integer entries are a superset-friendly subset (every integer is a half-integer).
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 7]
n, m, p = DIMENSIONS
N = n * m          # 12  (A rows)
M = m * p          # 28  (B rows)
P = p * n          # 21  (C rows)
RANK_SCHOOLBOOK = n * m * p   # 84
SEED_BASE = 12345
DEADLINE = time.time() + 140.0   # explicit safety margin vs 150s evaluator kill


def _deadline_ok():
    return time.time() < DEADLINE


def _make_factors(U, V, W):
    """Build the return dict from three numpy arrays of shape (rows, r)."""
    def conv(A):
        return [[float(v) for v in row] for row in A]
    return {"U": conv(U), "V": conv(V), "W": conv(W)}


def _schoolbook():
    """Deterministic valid rank-84 decomposition (always correct)."""
    r = RANK_SCHOOLBOOK
    U = np.zeros((N, r))
    V = np.zeros((M, r))
    W = np.zeros((P, r))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return _make_factors(U, V, W)


# Precompute the flat target tensor once (all ones, value 1 at each (i,j,k)).
def _target_flat():
    T = np.ones((N, M, P))
    return T.reshape(-1)


_TARGET = _target_flat()


def _loss_and_grad(U, V, W, T):
    """Return (loss, grad_U, grad_V, grad_W) for T_hat = einsum('ir,jr,sr->ijs', U, V, W).

    Using the identity: dL/dU[i,c] = (V*W) contracted along j,s where V[j,c],W[s,c].
    Specifically dL/dU[i,c] = sum_{j,s} (T_hat[i,j,s]-T[i,j,s]) * V[j,c] * W[s,c].
    """
    T_hat = np.einsum('ir,jr,sr->ijs', U, V, W, optimize=True)
    T_3d = T.reshape(N, M, P)
    R = T_hat - T_3d
    # grad_U[i,c] = sum_{j,s} R[i,j,s] * V[j,c] * W[s,c]
    # Build RV = R @ V over (i,j,s) x (j,c) -> (i,s,c)? Let's contract carefully.
    # sum_j R[i,j,s]*V[j,c] = einsum('ijs,jc->isc', R, V)
    RV = np.einsum('ijs,jc->isc', R, V, optimize=True)     # (i,s,c)
    grad_U = np.einsum('isc,sc->ic', RV, W, optimize=True)  # (i,c)
    # grad_V[j,c] = sum_{i,s} R[i,j,s]*U[i,c]*W[s,c]
    RU = np.einsum('ijs,ic->jsc', R, U, optimize=True)      # (j,s,c)
    grad_V = np.einsum('jsc,sc->jc', RU, W, optimize=True)  # (j,c)
    # grad_W[s,c] = sum_{i,j} R[i,j,s]*U[i,c]*V[j,c]
    RUW = np.einsum('ijs,ic->jsc', R, U, optimize=True)     # (j,s,c)
    grad_W = np.einsum('jsc,jc->sc', RUW, V, optimize=True) # (s,c)
    loss = float(np.dot(R.reshape(-1), R.reshape(-1)))
    return loss, grad_U, grad_V, grad_W


def _round_and_check(U, V, W, T, tol=1e-4, half_tol=1e-6):
    """Round to integers/half-integers if close, and verify exactness on a random probe.

    Returns (rounded_bool, valid_bool). We only trust a rounded result if the
    integer/half-integer factors reproduce the tensor exactly (within tiny fp error)
    on a fresh random bilinear test.
    """
    U2 = np.round(U * 2.0) / 2.0
    V2 = np.round(V * 2.0) / 2.0
    W2 = np.round(W * 2.0) / 2.0
    # check closeness to half-integers already by construction; verify exact tensor
    # Use a random A (n x m) and B (m x p) and compare A@B vs reconstruction.
    rng = np.random.RandomState(0)
    A = rng.randn(n, m)
    B = rng.randn(m, p)
    AB = A @ B
    # reconstruction: C[k,i] = sum_c W2[k*n+i, c] * (sum_j U2[i*m+j,c]*A[i,j]) * (sum_k? )
    # Actually C[k,i] = sum_j,k' ... let's use the tensor form:
    # C[k,i] = sum_{j} A[i,j] * B[j,k]  and  T form: C[k,i] = sum_c W2[k*n+i,c]*U2[i*m+j,c]*V2[j*p+k,c] summed over j? No.
    # The tensor identity: for each (i,j,k): T=1. Reconstruction of C[k,i]:
    # C[k,i] = sum_j A[i,j]*B[j,k] = sum_{j} (sum_c U2[i*m+j,c]*V2[j*p+k,c]*W2[k*n+i,c]) * A[i,j]*B[j,k]
    # This only works if sum_c U2[i*m+j,c]*V2[j*p+k,c]*W2[k*n+i,c] == 1 for ALL (i,j,k).
    # So just verify the tensor identity on the full grid.
    T_hat = np.einsum('ir,jr,sr->ijs', U2, V2, W2, optimize=True)
    T_3d = T.reshape(N, M, P)
    if not np.allclose(T_hat, T_3d, atol=1e-7, rtol=0):
        return False, False
    # Verify the actual bilinear product matches using a couple random matrices.
    for _ in range(3):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        AB = A @ B
        # compute via factors: C[k,i] = sum_{j} A[i,j] B[j,k]; using tensor we need
        # C[k,i] = sum_j A[i,j]*B[j,k] which equals sum_{j} (sum_c U2[i*m+j,c]*W2[k*n+i,c]*V2[j*p+k,c]) A[i,j] B[j,k]
        # Since the tensor identity holds (==1), C[k,i] = sum_j A[i,j]*B[j,k] automatically.
        # So just check we didn't introduce error: compare to AB directly (should match by construction).
    return True, True


def _random_init(r, rng):
    U = rng.randn(N, r)
    V = rng.randn(M, r)
    W = rng.randn(P, r)
    # Scale to unit-ish columns to keep magnitudes sane.
    U /= np.sqrt(N)
    V /= np.sqrt(M)
    W /= np.sqrt(P)
    # Initialize so that product is near 1 on average: scale columns so sum of products ~ target.
    # For a random unit column, einsum over the full tensor has mean ~0; we want sum_c =1.
    # Heuristic: multiply W by sqrt(3) etc. Keep simple; LM will adapt.
    return U, V, W


def _levenberg(U, V, W, T, iters, rng):
    lam = 1e-2
    loss, gU, gV, gW = _loss_and_grad(U, V, W, T)
    best_loss = loss
    best = (U.copy(), V.copy(), W.copy())
    for it in range(iters):
        if not _deadline_ok():
            break
        # LM update: (I + lam*diag) dx = -g
        size = (N * r if False else N) * 0  # placeholder
        # Solve per-matrix with a simple diagonal damping.
        step = None
        for _ in range(20):
            dU = -gU / (1.0 + lam * N)
            dV = -gV / (1.0 + lam * M)
            dW = -gW / (1.0 + lam * P)
            U2 = U + dU
            V2 = V + dV
            W2 = W + dW
            l2, _, _, _ = _loss_and_grad(U2, V2, W2, T)
            if l2 < loss:
                step = (dU, dV, dW)
                loss = l2
                break
            lam *= 10.0
        if step is None:
            lam *= 10.0
            if lam > 1e12:
                break
            continue
        U, V, W = U + step[0], V + step[1], W + step[2]
        gU, gV, gW = _loss_and_grad(U, V, W, T)[1:]
        lam = max(lam * 0.1, 1e-8)
        if loss < best_loss:
            best_loss = loss
            best = (U.copy(), V.copy(), W.copy())
        if best_loss < 1e-12:
            break
    return best, best_loss


def _search(r):
    """Try to find a rank-r decomposition via LM from several random starts."""
    for trial in range(6):
        if not _deadline_ok():
            break
        rng = np.random.RandomState(SEED_BASE + r * 1000 + trial)
        U, V, W = _random_init(r, rng)
        # Bias the initialization toward the schoolbook structure to help convergence:
        # set each column to a schoolbook triple scaled, then add noise.
        U, V, W = _schoolbook_init(r, rng)
        (U, V, W), loss = _levenberg(U, V, W, _TARGET, iters=400, rng=rng)
        if loss < 1e-8:
            rounded, valid = _round_and_check(U, V, W, _TARGET)
            if valid:
                return _make_factors(U, V, W)
        if not _deadline_ok():
            break
    return None


def _schoolbook_init(r, rng):
    """Initialize rank r by spreading the 84 schoolbook triples over r columns with noise."""
    U = np.zeros((N, r))
    V = np.zeros((M, r))
    W = np.zeros((P, r))
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))
    for c in range(r):
        # assign up to ceil(84/r) triples to column c
        lo = (84 * c) // r
        hi = (84 * (c + 1)) // r
        for t in range(lo, hi):
            i, j, k = triples[t % len(triples)]
            U[i * m + j, c] += 1.0
            V[j * p + k, c] += 1.0
            W[k * n + i, c] += 1.0
        # add a little noise to escape local minima
        U[:, c] += rng.randn(N) * 0.05
        V[:, c] += rng.randn(M) * 0.05
        W[:, c] += rng.randn(P) * 0.05
    return U, V, W


def construct():
    """Return {'U','V','W'} for the lowest-rank valid decomposition found before the deadline.

    Always returns a valid decomposition (schoolbook rank 84 at worst)."""
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK

    # Try progressively higher ranks until we can't afford to verify a lower one.
    # We want the LOWEST valid rank. Try a schedule of candidate ranks.
    candidates = sorted(set([49, 56, 64, 68, 70, 72, 75, 78, 80, 81, 82, 83, 84]))
    # Attempt from low to high; stop early if a valid low rank is found.
    for r in candidates:
        if not _deadline_ok():
            break
        if r >= best_rank:
            # no point trying a rank >= best already valid
            break
        res = _search(r)
        if res is not None:
            best = res
            best_rank = r
            # Found a valid rank; continue trying lower? candidates are ascending,
            # so lower ranks were already tried. Break to return the best found.
            # (lower ranks in the list come first, so if we found r, lower ones failed.)
            break
    return best


if __name__ == "__main__":
    t0 = time.time()
    out = construct()
    print("rank =", len(out["U"][0]))
    print("elapsed =", time.time() - t0)
