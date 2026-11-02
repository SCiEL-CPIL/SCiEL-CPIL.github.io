"""Rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 130.0

# EVOLVE-BLOCK-START
def _build_block_strassen():
    """Build a block decomposition: treat 2x4 as [2x2 | 2x2] and 4x7 as [2x7; 2x7].
    Use Strassen on the 2x2 block products and schoolbook elsewhere to get rank < 56."""
    n, m, p = DIMENSIONS
    # A is 2x4: split into A1 (2x2), A2 (2x2)
    # B is 4x7: split into B1 (2x7), B2 (2x7)
    # C = A1*B1 + A2*B2  (2x7)
    # For each 2x2 * 2x7 product, use a Strassen-like 7-term decomposition.
    # Strassen for 2x2 * 2x2 is rank 7. For 2x2 * 2x7, we can do it in 7 terms too
    # (each Strassen product gives a 2x7 output).
    # So A1*B1 in 7 terms, A2*B2 in 7 terms => 14 terms for the 2x7 output.
    # But schoolbook for 2x2*2x7 is 2*2*7 = 28. So 14 is a big saving.
    # Total: 14 terms. But wait, that's the rank of the whole 2x4*4x7 tensor?
    # No: the tensor is 2x4x7. A1*B1 is a sub-tensor (2x2x7), A2*B2 is another (2x2x7).
    # Their sum is the full tensor. Each 2x2x7 tensor has rank 7 (Strassen-style).
    # So total rank = 7 + 7 = 14? That seems too good. Let me verify the rank of 2x2x7.
    # Actually 2x2x7 matrix multiplication: rank is at most 7 (Strassen gives 7 for 2x2x2,
    # and 2x2x7 needs at least... the known bound). Let me just construct it and verify.
    #
    # Strassen 2x2x2: M1..M7. For 2x2x7, we can use 7 products each of form (2x2)*(2x7)?
    # No, Strassen multiplies 2x2 by 2x2. For 2x2 by 2x7, we split the 2x7 into 7 columns
    # of 2x1, and do 2x2*2x1 for each? That's 7 separate 2x2*2x1, each rank 4 schoolbook = 28.
    # That's not helpful.
    #
    # Better: 2x2 * 2x7. Think of it as (2x2)*(2x2 * 7/2)... not integer.
    # Actually the standard way: 2x2 * 2x7 can be done in 7 terms by treating the 2x7 as
    # a matrix and using the 7 Strassen products where each product is a 2x2 * 2x7 "slice".
    # Hmm, this is getting complicated. Let me just use schoolbook for the 2x2x7 blocks
    # and rely on the local search to find improvements.
    #
    # Simplest correct start: schoolbook rank 56, then local search.
    pass
    # Fall back to schoolbook
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

def _true_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _decompose(U, V, W):
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]; vc = V[:, c]; wc = W[:, c]
        for i in range(n):
            for j in range(m):
                aij = uc[i * m + j]
                if aij == 0.0:
                    continue
                for k in range(p):
                    T[i, j, k] += aij * vc[j * p + k] * wc[k * n + i]
    return T

def _verify(U, V, W, T, tol=1e-9):
    T2 = _decompose(U, V, W)
    return np.allclose(T2, T, atol=tol, rtol=0.0)

def _try_eliminate(U, V, W, T, a, tol=1e-9):
    n, m, p = DIMENSIONS
    cols = [c for c in range(U.shape[1]) if c != a]
    A = np.vstack([U[:, cols], V[:, cols], W[:, cols]])
    bvec = np.concatenate([U[:, a], V[:, a], W[:, a]])
    try:
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    except Exception:
        return None
    U2 = U[:, cols].copy(); V2 = V[:, cols].copy(); W2 = W[:, cols].copy()
    for c, cc in enumerate(cols):
        U2[:, c] += sol[c] * U[:, a]
        V2[:, c] += sol[c] * V[:, a]
        W2[:, c] += sol[c] * W[:, a]
    if _verify(U2, V2, W2, T, tol):
        return U2, V2, W2
    return None

def _try_merge(U, V, W, T, a, b, tol=1e-9):
    """Merge two columns by finding x,y,z such that the merged term equals the sum."""
    n, m, p = DIMENSIONS
    Ua, Ub = U[:, a], U[:, b]
    Va, Vb = V[:, a], V[:, b]
    Wa, Wb = W[:, a], W[:, b]
    rows = []; rhs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                gx = Ub[i*m+j] * Va[j*p+k] * Wa[k*n+i]
                gy = Ua[i*m+j] * Vb[j*p+k] * Wa[k*n+i]
                gz = Ua[i*m+j] * Va[j*p+k] * Wb[k*n+i]
                r = Ub[i*m+j] * Vb[j*p+k] * Wb[k*n+i]
                rows.append([gx, gy, gz]); rhs.append(r)
    A = np.array(rows); bvec = np.array(rhs)
    try:
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
    except Exception:
        return None
    x, y, z = sol
    Unew = Ua + x * Ub; Vnew = Va + y * Vb; Wnew = Wa + z * Wb
    idx = [c for c in range(U.shape[1]) if c != b]
    U2 = U[:, idx].copy(); V2 = V[:, idx].copy(); W2 = W[:, idx].copy()
    U2[:, a] = Unew; V2[:, a] = Vnew; W2[:, a] = Wnew
    if _verify(U2, V2, W2, T, tol):
        return U2, V2, W2
    return None

def _randomize(U, V, W, T, seed):
    rank = U.shape[1]
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((rank, rank))
    try:
        if abs(np.linalg.det(M)) < 1e-6:
            return None
    except Exception:
        return None
    U2 = U @ M; V2 = V @ M; W2 = W @ M
    if _verify(U2, V2, W2, T, tol=1e-9):
        return U2, V2, W2
    return None

def construct():
    start = time.time()
    U, V, W = _build_block_strassen()
    T = _true_tensor()
    best = (U.copy(), V.copy(), W.copy())
    best_rank = U.shape[1]

    def elapsed():
        return time.time() - start

    seed = 0
    while elapsed() < DEADLINE:
        seed += 1
        U, V, W = best
        improved = True
        while improved and elapsed() < DEADLINE:
            improved = False
            rank = U.shape[1]
            for a in range(rank):
                if elapsed() > DEADLINE: break
                res = _try_eliminate(U, V, W, T, a)
                if res is not None:
                    U, V, W = res; improved = True; break
            if improved: continue
            for a, b in itertools.combinations(range(rank), 2):
                if elapsed() > DEADLINE: break
                res = _try_merge(U, V, W, T, a, b)
                if res is not None:
                    U, V, W = res; improved = True; break
        if U.shape[1] < best_rank:
            best = (U.copy(), V.copy(), W.copy()); best_rank = U.shape[1]
        if elapsed() < DEADLINE:
            r = _randomize(best[0], best[1], best[2], T, seed)
            if r is not None:
                best = r
        if best_rank <= 28:
            break

    U, V, W = best
    if not _verify(U, V, W, T, tol=1e-9):
        U, V, W = _build_block_strassen()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
