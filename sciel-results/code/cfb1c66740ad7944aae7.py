"""Rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]
N, M, P = DIMENSIONS
NM, MP, PN = N*M, M*P, P*N


def _build_tensor():
    """Build the matmul tensor T[i,j,k] = 1 for all valid indices."""
    return np.ones((N, M, P), dtype=float)


def _decompose_error(U, V, W):
    """Compute squared error of the decomposition T = sum_c U_c ⊗ V_c ⊗ W_c."""
    R = U.shape[1]
    err = 0.0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0.0
                for c in range(R):
                    s += U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c]
                err += (s - 1.0)**2
    return err


def _to_frac(x, tol=1e-9):
    """Snap to a simple rational if close enough."""
    if abs(x) < tol:
        return 0.0
    for den in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
        num = round(x * den)
        if abs(x - num / den) < tol:
            if num == 0:
                return 0.0
            return float(num) / den
    return x


def _verify_exact(U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    R = len(U[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = Fraction(0, 1)
                for c in range(R):
                    s += Fraction(U[i*M+j][c]) * Fraction(V[j*P+k][c]) * Fraction(W[k*N+i][c])
                if s != 1:
                    return False
    return True


def _snap_and_verify(U, V, W):
    """Try to snap floating point values to simple rationals and verify."""
    U_r = [[_to_frac(x) for x in row] for row in U]
    V_r = [[_to_frac(x) for x in row] for row in V]
    W_r = [[_to_frac(x) for x in row] for row in W]
    if _verify_exact(U_r, V_r, W_r):
        return (U_r, V_r, W_r)
    return None


def _coordinate_descent(U, V, W, n_inner=50, deadline=None):
    """Perform coordinate descent to minimize decomposition error."""
    R = U.shape[1]
    for _ in range(n_inner):
        if deadline is not None and time.time() > deadline:
            break
        # Update U: for each (i,j), solve linear system
        for i in range(N):
            for j in range(M):
                idx = i * M + j
                # M[k, c] = V[j*P+k, c] * W[k*N+i, c]
                # b[k] = 1
                M_mat = np.zeros((P, R))
                b = np.ones(P)
                for k in range(P):
                    for c in range(R):
                        M_mat[k, c] = V[j*P+k, c] * W[k*N+i, c]
                try:
                    sol = np.linalg.lstsq(M_mat, b, rcond=None)[0]
                    U[idx, :] = sol
                except:
                    pass
        # Update V: for each (j,k), solve linear system
        for j in range(M):
            for k in range(P):
                idx = j * P + k
                # M[i, c] = U[i*M+j, c] * W[k*N+i, c]
                # b[i] = 1
                M_mat = np.zeros((N, R))
                b = np.ones(N)
                for i in range(N):
                    for c in range(R):
                        M_mat[i, c] = U[i*M+j, c] * W[k*N+i, c]
                try:
                    sol = np.linalg.lstsq(M_mat, b, rcond=None)[0]
                    V[idx, :] = sol
                except:
                    pass
        # Update W: for each (k,i), solve linear system
        for k in range(P):
            for i in range(N):
                idx = k * N + i
                # M[j, c] = U[i*M+j, c] * V[j*P+k, c]
                # b[j] = 1
                M_mat = np.zeros((M, R))
                b = np.ones(M)
                for j in range(M):
                    for c in range(R):
                        M_mat[j, c] = U[i*M+j, c] * V[j*P+k, c]
                try:
                    sol = np.linalg.lstsq(M_mat, b, rcond=None)[0]
                    W[idx, :] = sol
                except:
                    pass
    return U, V, W


def _try_rank_R(R, n_seeds=8, max_time=10.0, seed_base=0, deadline=None):
    """Try to find a rank-R decomposition using random initializations."""
    best_err = float('inf')
    best = None

    for seed in range(seed_base, seed_base + n_seeds):
        if deadline is not None and time.time() > deadline - 2:
            break

        rng = np.random.RandomState(seed)

        # Try multiple initialization strategies
        for init_scale in [0.5, 1.0, 2.0]:
            if deadline is not None and time.time() > deadline - 2:
                break

            U = rng.randn(NM, R) * init_scale
            V = rng.randn(MP, R) * init_scale
            W = rng.randn(PN, R) * init_scale

            # Run coordinate descent
            U, V, W = _coordinate_descent(U, V, W, n_inner=30, deadline=deadline)

            err = _decompose_error(U, V, W)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())

            if best_err < 1e-10:
                break

    if best is None:
        return None

    if best_err < 1e-8:
        result = _snap_and_verify(*best)
        if result is not None:
            return result

    return None


def _try_structured_rank(R, deadline=None):
    """Try structured initializations based on known patterns."""
    best_err = float('inf')
    best = None

    # Pattern: assign each column to a different (i,j,k) combination
    total_slots = N * M * P
    for offset in range(min(R, total_slots)):
        if deadline is not None and time.time() > deadline - 5:
            break

        U = np.zeros((NM, R))
        V = np.zeros((MP, R))
        W = np.zeros((PN, R))

        idx = offset % total_slots
        i = idx // (M * P)
        j = (idx // P) % M
        k = idx % P

        U[i*M+j, offset] = 1.0
        V[j*P+k, offset] = 1.0
        W[k*N+i, offset] = 1.0

        U, V, W = _coordinate_descent(U, V, W, n_inner=20, deadline=deadline)
        err = _decompose_error(U, V, W)

        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())

    if best is not None and best_err < 1e-8:
        result = _snap_and_verify(*best)
        if result is not None:
            return result

    return None


def _schoolbook():
    """Fallback: schoolbook (rank 18) decomposition."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1
                V[j * P + k][column] = 1
                W[k * N + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Try ranks from 12 up to 17 (skip 18 schoolbook)
    for R in range(12, 18):
        if time.time() > deadline - 10:
            break

        # First try structured initialization
        if time.time() < deadline - 15:
            res = _try_structured_rank(R, deadline=deadline)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}

        # Then try random coordinate descent with multiple seed bases
        for seed_base in [0, 50, 100, 200, 300, 500, 1000, 2000]:
            if time.time() > deadline - 5:
                break
            res = _try_rank_R(R, n_seeds=4, max_time=5.0, seed_base=seed_base, deadline=deadline)
            if res is not None:
                U, V, W = res
                return {"U": U, "V": V, "W": W}

    # Fallback to schoolbook
    return _schoolbook()
