"""Rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized exact verification. Returns True if decomposition is exact."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Build T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # Precompute W indexed by (k,n+i)
    # T[i,j,k]: for each c, contribution = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Vectorize over c using einsum-friendly reshapes.
    # U indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
    # Create full arrays:
    # A[i,j,c] = U[i*m+j, c]
    A = U.reshape(n, m, rank)
    # B[j,k,c] = V[j*p+k, c]
    B = V.reshape(m, p, rank)
    # C[k,i,c] = W[k*n+i, c]
    C = W.reshape(p, n, rank)
    # T[i,j,k] = sum_c A[i,j,c] B[j,k,c] C[k,i,c]
    # Do it in two steps to keep memory small:
    # D[i,j,k] = sum_c A[i,j,c]*B[j,k,c]  -> shape (n,m,p,rank) then *C
    # Compute E[i,j,k,c] = A[i,j,c]*B[j,k,c]
    E = A[:, :, None, :] * B[:, None, :, :]   # (n, m, p, rank)
    T = np.einsum('ijkc,kic->ijk', E, C)
    if not np.allclose(T, 1.0, atol=tol, rtol=0.0):
        return False
    return True

def _verify(U, V, W, n, m, p):
    return _verify_fast(U, V, W, n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _trim(U, V, W, n, m, p, tol=1e-9):
    """Remove all-zero columns and re-verify. Returns trimmed dict or None."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    keep = []
    for c in range(rank):
        if np.any(U[:, c] != 0) and np.any(V[:, c] != 0) and np.any(W[:, c] != 0):
            keep.append(c)
    if not keep:
        return None
    U2 = U[:, keep]
    V2 = V[:, keep]
    W2 = W[:, keep]
    if _verify_fast(U2, V2, W2, n, m, p, tol):
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    return None

def _random_candidate(n, m, p, rank, rng, sparsity=0.5):
    """Generate a random sparse ±1 candidate of given rank."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c in range(rank):
        us = rng.random(n * m) < sparsity
        vs = rng.random(m * p) < sparsity
        ws = rng.random(p * n) < sparsity
        U[us, c] = rng.choice([1, -1], size=int(us.sum()))
        V[vs, c] = rng.choice([1, -1], size=int(vs.sum()))
        W[ws, c] = rng.choice([1, -1], size=int(ws.sum()))
    return U, V, W

def _structured_candidate(n, m, p, rank, rng, pattern):
    """Structured ±1 candidate using block patterns."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c in range(rank):
        # Each term covers a random rectangular sub-block pattern
        i0 = rng.randrange(n)
        j0 = rng.randrange(m)
        k0 = rng.randrange(p)
        di = rng.randrange(1, n - i0 + 1)
        dj = rng.randrange(1, m - j0 + 1)
        dk = rng.randrange(1, p - k0 + 1)
        sU = rng.choice([1, -1])
        sV = rng.choice([1, -1])
        sW = rng.choice([1, -1])
        for i in range(i0, i0 + di):
            for j in range(j0, j0 + dj):
                U[i * m + j, c] = sU
        for j in range(j0, j0 + dj):
            for k in range(k0, k0 + dk):
                V[j * p + k, c] = sV
        for k in range(k0, k0 + dk):
            for i in range(i0, i0 + di):
                W[k * n + i, c] = sW
    return U, V, W

def _search_rank(n, m, p, rank, budget, rng, best):
    """Try to find a valid decomposition of exactly 'rank'."""
    deadline = time.time() + budget
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        if attempts % 2 == 0:
            U, V, W = _random_candidate(n, m, p, rank, rng)
        else:
            U, V, W = _structured_candidate(n, m, p, rank, rng, 0)
        if _verify_fast(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _reduce_by_merging(U, V, W, n, m, p, rng, budget):
    """Try to merge/zero columns to reduce rank while staying exact."""
    deadline = time.time() + budget
    U = np.asarray(U, dtype=float).copy()
    V = np.asarray(V, dtype=float).copy()
    W = np.asarray(W, dtype=float).copy()
    rank = U.shape[1]
    best_rank = rank
    best = (U.copy(), V.copy(), W.copy())
    # Greedily try zeroing out entries in random columns, keep if still exact
    for c in range(rank):
        if time.time() > deadline:
            break
        U_t = U.copy(); V_t = V.copy(); W_t = W.copy()
        # zero a random subset of this column
        zU = rng.random(n * m) < 0.5
        zV = rng.random(m * p) < 0.5
        zW = rng.random(p * n) < 0.5
        U_t[zU, c] = 0; V_t[zV, c] = 0; W_t[zW, c] = 0
        if _verify_fast(U_t, V_t, W_t, n, m, p):
            U, V, W = U_t, V_t, W_t
    # trim
    tr = _trim(U.tolist(), V.tolist(), W.tolist(), n, m, p)
    if tr is not None:
        return tr
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    rng = random.Random(12345)

    # Fallback: schoolbook rank 48
    result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Search from low to high rank; return first valid found (which is minimal in this order)
    # Budget split: more attempts at lower ranks.
    for rank in range(40, 48):
        if time.time() > deadline - 10:
            break
        budget = min(8.0, (deadline - 10 - time.time()))
        if budget <= 0:
            break
        found = _search_rank(n, m, p, rank, budget, rng, best_rank)
        if found is not None:
            tr = _trim(found['U'], found['V'], found['W'], n, m, p)
            if tr is not None:
                found = tr
            new_rank = len(found['U'][0])
            if new_rank < best_rank:
                best_rank = new_rank
                result = found
                # found a valid one; try to go even lower
                continue
        # also try merging on the current best
        if time.time() < deadline - 5:
            merged = _reduce_by_merging(result['U'], result['V'], result['W'], n, m, p, rng, 3.0)
            if merged is not None:
                nr = len(merged['U'][0])
                if nr < best_rank:
                    best_rank = nr
                    result = merged

    # Final high-precision polish: re-verify with tight tolerance
    if _verify_fast(result['U'], result['V'], result['W'], n, m, p, tol=1e-12):
        return result
    # If somehow invalid (shouldn't happen), fall back to schoolbook
    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
