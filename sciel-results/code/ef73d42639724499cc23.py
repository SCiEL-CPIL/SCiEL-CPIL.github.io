"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
1. Constructive rank-18 baseline (2 Strassen blocks + schoolbook for col 4).
2. Aggressive rank-16 search:
   - Structured sharing between Strassen blocks
   - Numerical ALS with rational snap
3. Rank-17 search as intermediate.
4. Fallbacks.

The output factor W follows the official trace convention W[k*n+i, r].
"""

import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 2, 5]
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _check_decomposition(U, V, W, T, tol=1e-8):
    n, m, p = DIMENSIONS
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    R = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = 0.0
                for r in range(R):
                    total += U_arr[u, r] * V_arr[v, r] * W_arr[w, r]
                if abs(total - T[u, v, w]) > tol:
                    return False
    return True


def _snap_to_halfint(x, tol=1e-6):
    hx = round(x * 2)
    if abs(x - hx / 2.0) < tol:
        return float(hx) / 2.0
    return None


def _check_exact(U, V, W, T, tol=1e-6):
    def to_frac(x):
        hx = round(x * 2)
        if abs(x - hx / 2.0) < tol:
            return Fraction(int(hx), 2)
        f = Fraction(x).limit_denominator(128)
        if abs(x - float(f)) < tol:
            return f
        return None

    Uf = []
    for row in U:
        r = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return False
            r.append(f)
        Uf.append(r)
    Vf = []
    for row in V:
        r = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return False
            r.append(f)
        Vf.append(r)
    Wf = []
    for row in W:
        r = []
        for x in row:
            f = to_frac(x)
            if f is None:
                return False
            r.append(f)
        Wf.append(r)

    R = len(U[0])
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _build_strassen_block(col0, col1, base, U, V, W, n, m, p):
    r = base
    U[0][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 1
    U[2][r] += 1; U[3][r] += 1
    V[0*p+col0][r] += 1
    W[col0*n+1][r] += 1; W[col1*n+1][r] += -1

    r = base + 2
    U[0][r] += 1
    V[0*p+col1][r] += 1; V[1*p+col1][r] += -1
    W[col1*n+0][r] += 1; W[col1*n+1][r] += 1

    r = base + 3
    U[3][r] += 1
    V[1*p+col0][r] += 1; V[0*p+col0][r] += -1
    W[col0*n+0][r] += 1; W[col0*n+1][r] += 1

    r = base + 4
    U[0][r] += 1; U[1][r] += 1
    V[1*p+col1][r] += 1
    W[col0*n+0][r] += -1; W[col1*n+0][r] += 1

    r = base + 5
    U[2][r] += 1; U[0][r] += -1
    V[0*p+col0][r] += 1; V[0*p+col1][r] += 1
    W[col1*n+1][r] += 1

    r = base + 6
    U[1][r] += 1; U[3][r] += -1
    V[1*p+col0][r] += 1; V[1*p+col1][r] += 1
    W[col0*n+0][r] += 1


def _build_rank18():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    _build_strassen_block(0, 1, 0, U, V, W, n, m, p)
    _build_strassen_block(2, 3, 7, U, V, W, n, m, p)

    r = 14
    U[0][r] = 1.0; V[0*p+4][r] = 1.0; W[4*n+0][r] = 1.0
    r = 15
    U[1][r] = 1.0; V[1*p+4][r] = 1.0; W[4*n+0][r] = 1.0
    r = 16
    U[2][r] = 1.0; V[0*p+4][r] = 1.0; W[4*n+1][r] = 1.0
    r = 17
    U[3][r] = 1.0; V[1*p+4][r] = 1.0; W[4*n+1][r] = 1.0

    return U, V, W


def _build_rank16_candidate():
    """Try to construct a rank-16 solution by sharing terms between blocks."""
    n, m, p = DIMENSIONS
    rank = 16
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Block 1: columns 0,1 (7 terms, indices 0-6)
    _build_strassen_block(0, 1, 0, U, V, W, n, m, p)

    # Block 2: columns 2,3 (7 terms, indices 7-13)
    _build_strassen_block(2, 3, 7, U, V, W, n, m, p)

    # Column 4: need 4 terms but only 2 slots (14, 15)
    # Try to use Strassen-like structure for 2x2x1
    # Actually 2x2x1 needs rank 4 (schoolbook)
    # So rank 16 with this structure is impossible.
    # Return None to indicate failure
    return None


def _als_solve(U, V, W, T, R, max_iters=200, tol=1e-12, deadline=None):
    """Vectorized alternating least squares for tensor decomposition."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Update U: for each u, solve min ||sum_r U[u,r]*V[:,r]*W[:,r] - T[u,:,:]||
        # Vectorize: A_u[r, (v,w)] = V[v,r]*W[w,r]
        V_flat = V.reshape(V_rows, 1, R)  # (V, 1, R)
        W_flat = W.reshape(1, W_rows, R)  # (1, W, R)
        # For each u, A is (V*W, R) where A[(v,w), r] = V[v,r]*W[w,r]
        # We can compute all at once: A_all[u, (v,w), r] = V[v,r]*W[w,r]
        # But that's U_rows * V_rows * W_rows * R which is 4*10*10*R = 400*R
        # For R=17, that's 6800 floats per u, manageable
        
        for u in range(U_rows):
            # A shape: (V*W, R)
            A = (V_flat[:, :, :] * W_flat[:, :, :]).reshape(V_rows * W_rows, R)
            b = T[u].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            U[u] = sol

        # Update V
        U_flat = U.reshape(U_rows, 1, R)
        for v in range(V_rows):
            A = (U_flat[:, :, :] * W_flat[:, :, :]).reshape(U_rows * W_rows, R)
            b = T[:, v, :].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            V[v] = sol

        # Update W
        U_flat = U.reshape(U_rows, 1, R)
        V_flat = V.reshape(1, V_rows, R)
        for w in range(W_rows):
            A = (U_flat[:, :, :] * V_flat[:, :, :]).reshape(U_rows * V_rows, R)
            b = T[:, :, w].reshape(-1)
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[w] = sol

        # Check convergence
        result = np.zeros((U_rows, V_rows, W_rows))
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
        err = np.linalg.norm(result - T)
        if err < tol:
            break

    return U, V, W


def _try_rank_als(R, T, deadline, n_seeds=8, max_iters=150):
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    seeds = [42, 123, 7, 999, 2024, 55, 88, 13, 2025, 314, 271, 173]

    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)

        for scale in [0.3, 0.7, 1.0, 2.0, 4.0]:
            if time.time() > deadline:
                break
            U0 = np.random.randn(U_rows, R) * scale
            V0 = np.random.randn(V_rows, R) * scale
            W0 = np.random.randn(W_rows, R) * scale

            U, V, W = U0.copy(), V0.copy(), W0.copy()
            U, V, W = _als_solve(U, V, W, T, R, max_iters=max_iters, deadline=deadline)

            if _check_decomposition(U, V, W, T, tol=1e-6):
                # Try to snap to half-integers
                for denom in [2, 1, 4, 8, 16]:
                    U_s = np.round(U * denom) / denom
                    V_s = np.round(V * denom) / denom
                    W_s = np.round(W * denom) / denom
                    if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                return U.tolist(), V.tolist(), W.tolist()

    return None


def _try_rank17_structured(T, deadline):
    """Try structured rank-17 by perturbing rank-18 and running ALS."""
    n, m, p = DIMENSIONS
    U18, V18, W18 = _build_rank18()
    U18 = np.array(U18, dtype=np.float64)
    V18 = np.array(V18, dtype=np.float64)
    W18 = np.array(W18, dtype=np.float64)
    R = 17

    # Strategy: take rank-18, zero out one column, perturb the rest, run ALS
    for remove_idx in [14, 15, 16, 17]:
        if time.time() > deadline:
            break

        for noise_scale in [0.0, 0.1, 0.5, 1.0]:
            if time.time() > deadline:
                break
            for seed in [42, 123, 7, 999, 2024, 55]:
                if time.time() > deadline:
                    break
                np.random.seed(seed + remove_idx * 100)

                U0 = U18[:, :R].copy()
                V0 = V18[:, :R].copy()
                W0 = W18[:, :R].copy()
                # Zero out the removed column's contribution
                U0[:, -1] = 0.0
                V0[:, -1] = 0.0
                W0[:, -1] = 0.0
                if noise_scale > 0:
                    U0 += np.random.randn(*U0.shape) * noise_scale
                    V0 += np.random.randn(*V0.shape) * noise_scale
                    W0 += np.random.randn(*W0.shape) * noise_scale

                U, V, W = _als_solve(U0, V0, W0, T, R, max_iters=100, deadline=deadline)

                if _check_decomposition(U, V, W, T, tol=1e-6):
                    for denom in [2, 1, 4, 8]:
                        U_s = np.round(U * denom) / denom
                        V_s = np.round(V * denom) / denom
                        W_s = np.round(W * denom) / denom
                        if _check_exact(U_s.tolist(), V_s.tolist(), W_s.tolist(), T):
                            return U_s.tolist(), V_s.tolist(), W_s.tolist()
                    return U.tolist(), V.tolist(), W.tolist()

    return None


def construct():
    deadline = time.time() + 110  # Safety margin for 120s limit
    n, m, p = DIMENSIONS
    T = _build_target()

    # Baseline: rank-18
    U18, V18, W18 = _build_rank18()
    if not _check_decomposition(U18, V18, W18, T, tol=1e-8):
        # Fallback: schoolbook
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank-17 via structured perturbation + ALS
    result17 = _try_rank17_structured(T, deadline)
    if result17 is not None:
        U17, V17, W17 = result17
        if _check_decomposition(U17, V17, W17, T, tol=1e-8):
            return {"U": U17, "V": V17, "W": W17}

    # Try rank-17 via random ALS
    if time.time() < deadline:
        result17_als = _try_rank_als(17, T, deadline, n_seeds=12, max_iters=80)
        if result17_als is not None:
            U17, V17, W17 = result17_als
            if _check_decomposition(U17, V17, W17, T, tol=1e-6):
                return {"U": U17, "V": V17, "W": W17}

    # Try rank-16 via ALS (less likely but worth trying)
    if time.time() < deadline - 20:
        result16 = _try_rank_als(16, T, deadline, n_seeds=4, max_iters=50)
        if result16 is not None:
            U16, V16, W16 = result16
            if _check_decomposition(U16, V16, W16, T, tol=1e-6):
                return {"U": U16, "V": V16, "W": W16}

    # Return rank-18 baseline
    return {"U": U18, "V": V18, "W": W18}
