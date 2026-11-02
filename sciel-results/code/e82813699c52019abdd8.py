"""Tensor decomposition search for 2x3x10 matrix multiplication.

Strategy:
  1. Build a rank-55 Strassen-based exact decomposition (base).
  2. Local search: perturb columns of U,V (linear combinations / random
     directions) and solve the resulting linear system for W. Accept any
     move that stays exact and reduces the score (rank, then sparsity).
  3. Attempt a rank-54 construction via random 54-column bases, solving
     for W and rounding to half-integers.
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]


def _build_target(n, m, p):
    """Build the matrix-multiplication tensor T (nm, mp, pn) with T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _check_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Exact check: for each (i,k), sum_r W[k*n+i,r] * U[i*m:..,r] outer V[..,k,r] == I_m."""
    R = U.shape[1]
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W_mat = W.reshape(p, n, R)
    I = np.eye(m)
    for i in range(n):
        for k in range(p):
            w = W_mat[k, i, :]
            M = (U_mat[i] * w[None, :]).T @ V_mat[:, k]
            if np.max(np.abs(M - I)) > tol:
                return False
    return True


def _solve_W(U, V, n, m, p):
    """Given U (nm,R), V (mp,R), solve the linear system for W (pn,R). Returns None if not exact."""
    R = U.shape[1]
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W = np.zeros((p * n, R))
    I = np.eye(m)
    for i in range(n):
        for k in range(p):
            # M_ik[j,l] = sum_r U[i,j,r] * W[k,i,r] * V[l,k,r] = I[j,l]
            # Flatten: for each (j,l): sum_r U[i,j,r]*V[l,k,r] * W[k,i,r] = I[j,l]
            # Build A (m*m, R)
            A = np.zeros((m * m, R))
            for j in range(m):
                for l in range(m):
                    A[j * m + l] = U_mat[i, j] * V_mat[l, k]
            b = I.reshape(-1)
            # Solve A^T w = b  (R x m^2, R) -> w (R,)
            try:
                w = np.linalg.lstsq(A.T, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            W[k * n + i] = w
    return W


def _sparsity_score(U, V, W, rank):
    """Lower is better: rank, then number of non-zero entries."""
    nz = (np.abs(U) > 1e-9).sum() + (np.abs(V) > 1e-9).sum() + (np.abs(W) > 1e-9).sum()
    return (rank, int(nz))


def _round_half(X):
    return np.round(X * 2.0) / 2.0


def _build_rank55(n, m, p):
    """Strassen-based rank-55 exact decomposition for 2x3x10."""
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    # Part A: j = 0 (full schoolbook over i, k) -> 2*10 = 20 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Part B: j in {1,2}, pair up the p=10 columns into 5 pairs, apply Strassen per pair
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = (a11 + a22)(b11 + b22)
        U[u_11, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1; V[v_22, col] += 1
        W[w_00, col] += 1; W[w_11, col] += 1
        col += 1

        # M2 = (a21 + a22) b11
        U[u_21, col] += 1; U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1; W[w_11, col] += -1
        col += 1

        # M3 = a11 (b12 - b22)
        U[u_11, col] += 1
        V[v_12, col] += 1; V[v_22, col] += -1
        W[w_01, col] += 1
        col += 1

        # M4 = a12 b21
        U[u_12, col] += 1
        V[v_21, col] += 1
        W[w_10, col] += 1
        col += 1

        # M5 = (a11 - a22) b21
        U[u_11, col] += 1; U[u_22, col] += -1
        V[v_21, col] += 1
        W[w_10, col] += 1; W[w_11, col] += -1
        col += 1

        # M6 = (a12 - a11)(b21 + b11)
        U[u_12, col] += 1; U[u_11, col] += -1
        V[v_21, col] += 1; V[v_11, col] += 1
        W[w_10, col] += 1
        col += 1

        # M7 = (a21 - a22)(b12 + b22)
        U[u_21, col] += 1; U[u_22, col] += -1
        V[v_12, col] += 1; V[v_22, col] += 1
        W[w_01, col] += 1
        col += 1

    assert col == rank, (col, rank)
    return U, V, W


def _try_rank54(n, m, p, attempts=400):
    """Try to find a rank-54 decomposition by random bases + solve for W + round."""
    R = 54
    for a in range(attempts):
        np.random.seed(1000 + a)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = _solve_W(U, V, n, m, p)
        if W is None:
            continue
        # Round to half-integers and re-check
        U_r, V_r, W_r = _round_half(U), _round_half(V), _round_half(W)
        if _check_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-9):
            return U_r, V_r, W_r
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115.0  # safety margin under 150s

    U, V, W = _build_rank55(n, m, p)
    best = (U.copy(), V.copy(), W.copy())
    best_score = _sparsity_score(U, V, W, U.shape[1])

    # Verify base is exact
    assert _check_decomposition(U, V, W, n, m, p, tol=1e-7), "rank-55 base not exact"

    random.seed(0)

    # ---- Local search on rank-55: perturb columns, re-solve W, accept if exact + better ----
    R = U.shape[1]
    while time.time() < deadline - 10.0:
        # Pick a random pair of columns and a random mixing matrix, or a random direction
        mode = random.random()
        U_c = U.copy()
        V_c = V.copy()
        if mode < 0.5:
            # Linear combination of two random columns in U (and matching in V)
            a, b = random.sample(range(R), 2)
            t = random.uniform(-0.5, 0.5)
            U_c[:, a] = (1 + t) * U[:, a] + t * U[:, b]
            U_c[:, b] = U[:, b]
            V_c[:, a] = (1 + t) * V[:, a] + t * V[:, b]
            V_c[:, b] = V[:, b]
        else:
            # Add small random direction to one column of U and V
            a = random.randrange(R)
            eps = random.uniform(0.05, 0.3)
            U_c[:, a] += eps * np.random.randn(n * m)
            V_c[:, a] += eps * np.random.randn(m * p)

        W_c = _solve_W(U_c, V_c, n, m, p)
        if W_c is None:
            continue
        if not _check_decomposition(U_c, V_c, W_c, n, m, p, tol=1e-8):
            continue
        sc = _sparsity_score(U_c, V_c, W_c, R)
        if sc < best_score:
            best = (U_c.copy(), V_c.copy(), W_c.copy())
            best_score = sc

    # ---- Attempt rank-54 (only if time remains) ----
    if time.time() < deadline - 5.0:
        r54 = _try_rank54(n, m, p, attempts=200)
        if r54 is not None:
            U54, V54, W54 = r54
            if _check_decomposition(U54, V54, W54, n, m, p, tol=1e-9):
                if _sparsity_score(U54, V54, W54, 54) < best_score:
                    best = (U54, V54, W54)
                    best_score = _sparsity_score(U54, V54, W54, 54)

    # ---- Final polish: try rounding the best found to half-integers ----
    Ub, Vb, Wb = best
    Ur, Vr, Wr = _round_half(Ub), _round_half(Vb), _round_half(Wb)
    if _check_decomposition(Ur, Vr, Wr, n, m, p, tol=1e-9):
        if _sparsity_score(Ur, Vr, Wr, Ur.shape[1]) <= best_score:
            best = (Ur, Vr, Wr)

    Ub, Vb, Wb = best
    U_list = [[float(x) for x in row] for row in Ub]
    V_list = [[float(x) for x in row] for row in Vb]
    W_list = [[float(x) for x in row] for row in Wb]
    return {"U": U_list, "V": V_list, "W": W_list}
