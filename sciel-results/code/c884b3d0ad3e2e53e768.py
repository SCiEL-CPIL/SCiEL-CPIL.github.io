import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # 1. Construct Schoolbook Decomposition (Rank n*m*p = 48)
    rank_sb = n * m * p
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb

    # 2. Structured search: prune from schoolbook, then local optimize
    # Try decreasing ranks. For each, try multiple strategies.
    for rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 8:
            break

        found = _try_find_rank(n, m, p, rank,
                               max_restarts=25,
                               time_limit=9,
                               deadline=deadline)
        if found is not None:
            U, V, W = found
            best_result = (U, V, W)
            best_rank = rank
            break

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_find_rank(n, m, p, rank, max_restarts=10, time_limit=10, deadline=None):
    start_time = time.time()

    for restart in range(max_restarts):
        if time.time() > start_time + time_limit:
            break
        if deadline is not None and time.time() > deadline - 5:
            break

        # Strategy A: prune from schoolbook (warm start)
        # Strategy B: structured random init
        if restart % 2 == 0:
            U, V, W = _init_prune(n, m, p, rank)
        else:
            U, V, W = _init_random(n, m, p, rank)

        # ALS iterations
        U, V, W, max_err = _als(U, V, W, n, m, p,
                                 iters=60,
                                 start_time=start_time,
                                 time_limit=time_limit)

        # Check
        if max_err < 1e-6:
            # Polish with Newton
            U2, V2, W2 = _polish(U, V, W, n, m, p,
                                  time_left=time_limit - (time.time() - start_time))
            if U2 is not None:
                return U2, V2, W2
            else:
                # Try direct snap
                snapped, ok = _snap_all(U, V, W, 1e-4)
                if ok and _check_exact(snapped[0], snapped[1], snapped[2], n, m, p):
                    return snapped

    return None


def _init_prune(n, m, p, rank):
    """Start from schoolbook, keep 'rank' random terms, re-solve."""
    nmp = n * m * p
    U, V, W = _schoolbook(n, m, p)

    # Choose 'rank' random columns to keep
    cols = random.sample(range(nmp), rank)
    U = U[:, cols]
    V = V[:, cols]
    W = W[:, cols]

    # Add small perturbation to allow re-solvability
    U += np.random.randn(n * m, rank) * 0.05
    V += np.random.randn(m * p, rank) * 0.05
    W += np.random.randn(p * n, rank) * 0.05
    return U, V, W


def _init_random(n, m, p, rank):
    """Structured random init with some schoolbook structure."""
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r in range(rank):
        # Each rank-1 term should "cover" some (i,j,k) pattern
        # Assign a random (i,j,k) cluster to this term
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Probability this term contributes to (i,j,k)
                    if random.random() < 0.25:
                        U[i * m + j, r] = random.choice([-1.0, 1.0])
                        V[j * p + k, r] = random.choice([-1.0, 1.0])
                        W[k * n + i, r] = random.choice([-1.0, 1.0])

    # Add noise to break symmetry
    U += np.random.randn(n * m, rank) * 0.2
    V += np.random.randn(m * p, rank) * 0.2
    W += np.random.randn(p * n, rank) * 0.2
    return U, V, W


def _als(U, V, W, n, m, p, iters=60, start_time=None, time_limit=10):
    """Alternating least squares for tensor decomposition."""
    max_err = float('inf')
    prev_err = float('inf')

    for it in range(iters):
        if start_time is not None and time.time() > start_time + time_limit:
            break

        # Update U: for each (i,j), solve least squares
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # V_part: rows j*p .. j*p+p-1  (shape p x rank)
                # W_part: rows k*n+i for k=0..p-1 (shape p x rank)
                V_part = V[j * p:(j + 1) * p, :]
                W_part = W[np.arange(p) * n + i, :]
                A = V_part * W_part  # (p, rank)
                b = np.ones(p)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    U_new[idx, :] = x
                except Exception:
                    U_new[idx, :] = U[idx, :]
        U = U_new

        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                U_part = U[np.arange(n) * m + j, :]  # (n, rank)
                W_part = W[k * n:(k + 1) * n, :]     # (n, rank)
                A = U_part * W_part
                b = np.ones(n)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    V_new[idx, :] = x
                except Exception:
                    V_new[idx, :] = V[idx, :]
        V = V_new

        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                U_part = U[i * m:(i + 1) * m, :]     # (m, rank)
                V_part = V[np.arange(m) * p + k, :]  # (m, rank)
                A = U_part * V_part
                b = np.ones(m)
                try:
                    x = np.linalg.lstsq(A, b, rcond=None)[0]
                    W_new[idx, :] = x
                except Exception:
                    W_new[idx, :] = W[idx, :]
        W = W_new

        # Compute max error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    e = abs(val - 1.0)
                    if e > max_err:
                        max_err = e

        if max_err < 1e-8:
            break
        if abs(prev_err - max_err) < 1e-11:
            break
        prev_err = max_err

    return U, V, W, max_err


def _polish(U, V, W, n, m, p, time_left=5):
    """Newton-style polish to drive error to machine precision, then snap."""
    if time_left < 0.5:
        return None, None, None

    start = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]

    for it in range(30):
        if time.time() - start > time_left:
            break

        # Gauss-Seidel Newton updates
        # Update U
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                res = np.zeros(p)
                for k in range(p):
                    val = np.dot(U[idx, :], V[j * p + k, :] * W[k * n + i, :])
                    res[k] = 1.0 - val
                V_part = V[j * p:(j + 1) * p, :]
                W_part = W[np.arange(p) * n + i, :]
                A = V_part * W_part
                try:
                    delta = np.linalg.solve(A, res)
                    U[idx, :] += delta
                except Exception:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                res = np.zeros(n)
                for i in range(n):
                    val = np.dot(U[i * m + j, :], V[idx, :] * W[k * n + i, :])
                    res[i] = 1.0 - val
                U_part = U[np.arange(n) * m + j, :]
                W_part = W[k * n:(k + 1) * n, :]
                A = U_part * W_part
                try:
                    delta = np.linalg.solve(A, res)
                    V[idx, :] += delta
                except Exception:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                res = np.zeros(m)
                for j in range(m):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[idx, :])
                    res[j] = 1.0 - val
                U_part = U[i * m:(i + 1) * m, :]
                V_part = V[np.arange(m) * p + k, :]
                A = U_part * V_part
                try:
                    delta = np.linalg.solve(A, res)
                    W[idx, :] += delta
                except Exception:
                    pass

        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    e = abs(val - 1.0)
                    if e > max_err:
                        max_err = e
        if max_err < 1e-10:
            break

    # Final snap to half-integers
    snapped, ok = _snap_all(U, V, W, 1e-3)
    if ok and _check_exact(snapped[0], snapped[1], snapped[2], n, m, p):
        return snapped
    return None, None, None


def _snap_all(U, V, W, tol=1e-6):
    """Snap all three to half-integers if close."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    ok = (np.max(np.abs(U - U_s)) < tol and
          np.max(np.abs(V - V_s)) < tol and
          np.max(np.abs(W - W_s)) < tol)
    return (U_s, V_s, W_s), ok


def _check_exact(U, V, W, n, m, p, tol=1e-9):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
