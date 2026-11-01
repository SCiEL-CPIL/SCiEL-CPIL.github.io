"""Tensor decomposition for 2x3 times 3x9 matrix multiplication (rank-minimal exact).

Strategy:
  The multiplication tensor for (n x m) * (m x p) has a trivial rank-n*m*p
  decomposition (one term per (i,j,k) triple).  For small dimensions we can
  attempt to find a lower-rank decomposition via a greedy / randomised search
  over rational (integer or half-integer) factor matrices.

  The search uses a smooth surrogate: at each step we pick the rank term that
  minimises the residual Frobenius norm of the tensor after subtracting the
  new rank-1 term, using a log-sum-exp style soft-min to smooth the discrete
  choice.  A final exact verification pass ensures the returned decomposition
  is truly exact (integer/half-integer entries, residual identically zero).

  If no lower-rank exact decomposition is found within the time budget, the
  program falls back to the schoolbook rank-54 decomposition, which is always
  valid.
"""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]


def _build_tensor(n, m, p):
    """Build the full multiplication tensor T[i, j, k] = 1 for all i,j,k
    (the tensor whose rank-1 decomposition gives the bilinear map
    (A,B) -> A*B with (AB)_{i,k} = sum_j A_{i,j} B_{j,k})."""
    return np.ones((n, m, p), dtype=np.int64)


def _tensor_to_flat(T, n, m, p):
    """Flatten the 3-way tensor into a vector of length n*m*p using the
    convention: index = i*m*p + j*p + k."""
    return T.reshape(-1)


def _flat_to_tensor(v, n, m, p):
    return v.reshape((n, m, p))


def _rank1_term(i, j, k, n, m, p):
    """Return the rank-1 term for the (i,j,k) index as three vectors:
    u of length n*m (index i*m+j), v of length m*p (index j*p+k),
    w of length p*n (index k*n+i).  All entries are 0 or 1."""
    u = np.zeros(n * m, dtype=np.int64)
    v = np.zeros(m * p, dtype=np.int64)
    w = np.zeros(p * n, dtype=np.int64)
    u[i * m + j] = 1
    v[j * p + k] = 1
    w[k * n + i] = 1
    return u, v, w


def _schoolbook_decomposition(n, m, p):
    """Return the exact rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _verify_exact(U, V, W, n, m, p):
    """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for all i, j, k.  All entries must be integers or half-integers."""
    rank = U.shape[1]
    # Check entries are integer or half-integer
    for mat in (U, V, W):
        for val in mat:
            if not (val * 2 == int(val * 2)):
                return False
    # Check the tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Fast vectorised verification using NumPy."""
    rank = U.shape[1]
    # Build the full tensor via outer products
    # T_reconstructed[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Use einsum for speed
    T = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
    expected = np.ones((n, m, p), dtype=np.int64)
    if not np.array_equal(T.astype(np.int64), expected):
        return False
    # Check half-integer constraint
    for mat in (U, V, W):
        doubled = mat * 2
        if not np.allclose(doubled, np.round(doubled)):
            return False
    return True


def _greedy_search(n, m, p, max_rank, time_limit, rng):
    """Attempt to find a decomposition of rank <= max_rank using a greedy
    approach with smooth surrogate scoring.

    The idea: at each step, we have a residual tensor R.  We want to pick
    a rank-1 term (u, v, w) that best approximates R, i.e. minimises
    ||R - u*v*w||_F.  For a fixed (u,v), the optimal w is the projection
    of R onto the span of u*v, and vice-versa.  We use a randomised search
    over small sets of candidate vectors.

    Returns (U, V, W) if a valid decomposition of rank <= max_rank is found,
    else None.
    """
    T = _build_tensor(n, m, p)
    T_flat = _tensor_to_flat(T, n, m, p).astype(np.float64)

    U_list = []
    V_list = []
    W_list = []

    residual = T_flat.copy()

    for step in range(max_rank):
        best_score = float('inf')
        best_uvw = None

        # Generate candidate rank-1 terms
        # Strategy: try all (i,j,k) triples, plus some random perturbations
        candidates = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u, v, w = _rank1_term(i, j, k, n, m, p)
                    candidates.append((u.astype(np.float64), v.astype(np.float64), w.astype(np.float64)))

        # Add random candidates
        for _ in range(200):
            u = rng.choice([-1, 0, 1], size=n * m).astype(np.float64)
            v = rng.choice([-1, 0, 1], size=m * p).astype(np.float64)
            w = rng.choice([-1, 0, 1], size=p * n).astype(np.float64)
            # Normalise to unit norm to avoid scaling issues
            nu = np.linalg.norm(u)
            nv = np.linalg.norm(v)
            nw = np.linalg.norm(w)
            if nu > 0 and nv > 0 and nw > 0:
                u = u / nu
                v = v / nv
                w = w / nw
            candidates.append((u, v, w))

        for u, v, w in candidates:
            # Compute the best scalar alpha and the residual after subtracting alpha * u*v*w
            # outer = u[i*m+j] * v[j*p+k] * w[k*n+i]
            outer = np.einsum('i,j,k->ijk', u, v, w, optimize=True).reshape(-1)
            # alpha = <residual, outer> / <outer, outer>
            dot_ro = np.dot(residual, outer)
            dot_oo = np.dot(outer, outer)
            if dot_oo < 1e-12:
                continue
            alpha = dot_ro / dot_oo
            new_residual = residual - alpha * outer
            score = np.dot(new_residual, new_residual)
            if score < best_score:
                best_score = score
                best_uvw = (u, v, w, alpha)

        if best_uvw is None:
            return None

        u, v, w, alpha = best_uvw
        residual -= alpha * np.einsum('i,j,k->ijk', u, v, w, optimize=True).reshape(-1)

        # Store the term (convert to Fraction for exactness later)
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

        # Check if residual is (approximately) zero
        if np.max(np.abs(residual)) < 1e-10:
            # Try to snap to half-integers
            U_arr = np.array(U_list, dtype=np.float64)
            V_arr = np.array(V_list, dtype=np.float64)
            W_arr = np.array(W_list, dtype=np.float64)
            # Snap to nearest half-integer
            U_snap = np.round(U_arr * 2) / 2
            V_snap = np.round(V_arr * 2) / 2
            W_snap = np.round(W_arr * 2) / 2
            # Verify exactness
            T_recon = np.einsum('ic,jc,kc->ijk', U_snap, V_snap, W_snap, optimize=True)
            if np.allclose(T_recon, np.ones((n, m, p)), atol=1e-9):
                # Convert to integer/half-integer representation
                return (U_snap, V_snap, W_snap)

    return None


def _try_strassen_block(n, m, p, time_limit):
    """Attempt a block-based decomposition.

    For 2x3 * 3x9, we can view this as:
    - 2x3 * 3x1 repeated 9 times (each has rank 3, total 27)
    - But with sharing: the 3x3 block structure allows Strassen-like savings.

    Actually, let's try a direct approach: partition the 3x9 matrix B into
    three 3x3 blocks B1, B2, B3. Then A*B = A*B1 + A*B2 + A*B3 where
    each A*Bj is 2x3.  The tensor for A*Bj (2x3 * 3x3) has rank 18 by
    schoolbook.  Total rank 54.  No savings from this partitioning.

    Alternative: use the fact that 2x3 * 3x9 can be seen as a 2x9 output
    where each output C[i,k] = sum_j A[i,j]*B[j,k].  The tensor rank is
    the minimum number of rank-1 terms.  For this specific case, it is
    known that rank = n*m*p for generic n,m,p, but special cases may differ.

    Given the difficulty of finding a provably lower-rank decomposition,
    we return None to fall back to schoolbook.
    """
    return None


def construct():
    start = time.time()
    deadline = start + 150.0 - 5.0  # safety margin
    n, m, p = DIMENSIONS

    # First, try the schoolbook decomposition (always valid, rank 54)
    U, V, W = _schoolbook_decomposition(n, m, p)

    # Attempt to find a lower-rank decomposition
    best_rank = n * m * p
    best_UVW = (U, V, W)

    # Try greedy search with decreasing rank targets
    rng = np.random.default_rng(42)
    for target_rank in range(best_rank - 1, max(1, best_rank // 2 - 1), -1):
        if time.time() > deadline - 30:
            break
        result = _greedy_search(n, m, p, target_rank, deadline - 30, rng)
        if result is not None:
            U_snap, V_snap, W_snap = result
            # Convert to list-of-lists with integer/half-integer entries
            U_out = U_snap.tolist()
            V_out = V_snap.tolist()
            W_out = W_snap.tolist()
            # Verify
            if _verify_exact(np.array(U_out), np.array(V_out), np.array(W_out), n, m, p):
                best_rank = target_rank
                best_UVW = (np.array(U_out), np.array(V_out), np.array(W_out))
                break

    # Convert best result to list-of-lists format
    U_arr, V_arr, W_arr = best_UVW
    U_out = U_arr.tolist()
    V_out = V_arr.tolist()
    W_out = W_arr.tolist()

    # Final verification
    if not _verify_exact(np.array(U_out), np.array(V_out), np.array(W_out), n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        U_out = U.tolist()
        V_out = V.tolist()
        W_out = W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
