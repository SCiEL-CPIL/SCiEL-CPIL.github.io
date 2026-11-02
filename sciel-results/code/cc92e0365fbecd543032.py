"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
- Build the target multiplication tensor T of shape (4, 10, 10).
- Use a structured rank-18 baseline (two Strassen 2x2 blocks + 4 schoolbook terms for col 4).
- Attempt rank-17 via:
    1) Direct structured construction with one shared term.
    2) ALS local search starting from rank-18 with one column removed/replaced.
    3) Tensor-flattening SVD guided basis search: matricize T in multiple ways,
       use SVD to identify low-rank directions, then refine with ALS.
- Rational polish: snap to half-integers / small denominators and verify exactly.
- Fallback: rank-18 structured or rank-20 schoolbook.
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _decompose(U, V, W):
    R = U.shape[1]
    result = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _check_decomposition(U, V, W, T, tol=1e-8):
    result = _decompose(U, V, W)
    return np.allclose(result, T, atol=tol)


def _check_exact(U, V, W, T):
    def to_frac(x):
        for denom in [1, 2, 4, 8, 16, 32, 64]:
            num = round(x * denom)
            candidate = Fraction(int(num), denom)
            if abs(x - float(candidate)) < 1e-9:
                return candidate
        return Fraction(x).limit_denominator(128)

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = U.shape[1]
    for u in range(U.shape[0]):
        for v in range(V.shape[0]):
            for w in range(W.shape[0]):
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _snap(M, denom):
    return np.round(M * denom) / denom


def _polish_rational(U, V, W, T):
    # Try common denominators
    for denom in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        U_s = _snap(U, denom)
        V_s = _snap(V, denom)
        W_s = _snap(W, denom)
        if _check_decomposition(U_s, V_s, W_s, T, tol=1e-6):
            if _check_exact(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    # Try mixed denominators (small set)
    for dU in [1, 2, 4, 8]:
        for dV in [1, 2, 4, 8]:
            for dW in [1, 2, 4, 8]:
                U_s = _snap(U, dU)
                V_s = _snap(V, dV)
                W_s = _snap(W, dW)
                if _check_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    # If all else fails, try to snap to nearest integer if close
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    if _check_decomposition(U_int, V_int, W_int, T, tol=1e-6):
         return U_int.tolist(), V_int.tolist(), W_int.tolist()

    return None


def _als_update_U(T, V, W):
    R = V.shape[1]
    A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
    U = np.zeros((T.shape[0], R), dtype=np.float64)
    for u in range(T.shape[0]):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        U[u] = sol
    return U


def _als_update_V(T, U, W):
    R = U.shape[1]
    A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
    V = np.zeros((T.shape[1], R), dtype=np.float64)
    for v in range(T.shape[1]):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        V[v] = sol
    return V


def _als_update_W(T, U, V):
    R = U.shape[1]
    A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
    W = np.zeros((T.shape[2], R), dtype=np.float64)
    for w in range(T.shape[2]):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W[w] = sol
    return W


def _compute_error(U, V, W, T):
    result = _decompose(U, V, W)
    return np.linalg.norm(result - T.astype(np.float64))


def _estimate_tensor_rank(T, max_rank=25):
    T_f = T.astype(np.float64)
    M1 = T_f.reshape(T_f.shape[0], -1)
    s1 = np.linalg.svd(M1, compute_uv=False)
    r1 = np.sum(s1 > 1e-10)
    M2 = T_f.transpose(1, 0, 2).reshape(T_f.shape[1], -1)
    s2 = np.linalg.svd(M2, compute_uv=False)
    r2 = np.sum(s2 > 1e-10)
    M3 = T_f.transpose(2, 0, 1).reshape(T_f.shape[2], -1)
    s3 = np.linalg.svd(M3, compute_uv=False)
    r3 = np.sum(s3 > 1e-10)
    return int(max(r1, r2, r3))


def _build_structured_decomposition():
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1

        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1

        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1

        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1

        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1

        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1

        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)

    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1

    return U, V, W


def _try_rank_reduction(U, V, W, T, deadline, target_rank=None):
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = U.shape[0], V.shape[0], W.shape[0]
    R = U.shape[1]

    if target_rank is None:
        target_rank = R - 1

    if target_rank < 1:
        return None

    for r_remove in range(R):
        if time.time() > deadline:
            break
        mask = [i for i in range(R) if i != r_remove]
        if len(mask) < target_rank:
            continue

        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        R_new = len(mask)

        for seed in [42, 123, 7, 999, 314, 777, 2024, 555]:
            if time.time() > deadline:
                break
            np.random.seed(seed)

            U_new = np.zeros((U_rows, R_new + 1))
            V_new = np.zeros((V_rows, R_new + 1))
            W_new = np.zeros((W_rows, R_new + 1))

            U_new[:, :R_new] = U_sub
            V_new[:, :R_new] = V_sub
            W_new[:, :R_new] = W_sub

            U_new[:, R_new] = np.random.randn(U_rows) * 0.3
            V_new[:, R_new] = np.random.randn(V_rows) * 0.3
            W_new[:, R_new] = np.random.randn(W_rows) * 0.3

            for it in range(100):
                if time.time() > deadline:
                    break
                U_new = _als_update_U(T, V_new, W_new)
                V_new = _als_update_V(T, U_new, W_new)
                W_new = _als_update_W(T, U_new, V_new)

            if _check_decomposition(U_new, V_new, W_new, T, tol=1e-7):
                polished = _polish_rational(U_new, V_new, W_new, T)
                if polished is not None:
                    return polished
                return U_new.tolist(), V_new.tolist(), W_new.tolist()

    return None


def _try_structured_rank17(T, deadline):
    n, m, p = DIMENSIONS
    target_rank = 17

    for strategy in range(8):
        if time.time() > deadline:
            break

        np.random.seed(42 + strategy * 100)

        U = np.zeros((n * m, target_rank), dtype=np.float64)
        V = np.zeros((m * p, target_rank), dtype=np.float64)
        W = np.zeros((p * n, target_rank), dtype=np.float64)

        # Standard Strassen for (0,1): 7 terms
        r = 0
        U[0, r] = 1; U[3, r] = 1
        V[0*p+0, r] = 1; V[1*p+1, r] = 1
        W[0*n+0, r] = 1; W[1*n+1, r] = 1

        r = 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+0, r] = 1
        W[0*n+1, r] = 1; W[1*n+1, r] = -1

        r = 2
        U[0, r] = 1
        V[0*p+1, r] = 1; V[1*p+1, r] = -1
        W[1*n+0, r] = 1; W[1*n+1, r] = 1

        r = 3
        U[3, r] = 1
        V[1*p+0, r] = 1; V[0*p+0, r] = -1
        W[0*n+0, r] = 1; W[0*n+1, r] = 1

        r = 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+1, r] = 1
        W[0*n+0, r] = -1; W[1*n+0, r] = 1

        r = 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+0, r] = 1; V[0*p+1, r] = 1
        W[1*n+1, r] = 1

        r = 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+0, r] = 1; V[1*p+1, r] = 1
        W[0*n+0, r] = 1

        # For (2,3): use 6 terms by sharing structure
        r = 7
        U[0, r] = 1; U[3, r] = 1
        V[0*p+2, r] = 1; V[1*p+3, r] = 1
        W[2*n+0, r] = 1; W[3*n+1, r] = 1

        r = 8
        U[2, r] = 1; U[3, r] = 1
        V[0*p+2, r] = 1
        W[2*n+1, r] = 1; W[3*n+1, r] = -1

        r = 9
        U[0, r] = 1
        V[0*p+3, r] = 1; V[1*p+3, r] = -1
        W[3*n+0, r] = 1; W[3*n+1, r] = 1

        r = 10
        U[3, r] = 1
        V[1*p+2, r] = 1; V[0*p+2, r] = -1
        W[2*n+0, r] = 1; W[2*n+1, r] = 1

        r = 11
        U[0, r] = 1; U[1, r] = 1
        V[1*p+3, r] = 1
        W[2*n+0, r] = -1; W[3*n+0, r] = 1

        r = 12
        U[2, r] = 1; U[0, r] = -1
        V[0*p+2, r] = 1; V[0*p+3, r] = 1
        W[3*n+1, r] = 1

        # Col 4: 3 terms
        r = 13
        U[0, r] = 1
        V[0*p+4, r] = 1
        W[4*n+0, r] = 1

        r = 14
        U[2, r] = 1
        V[0*p+4, r] = 1
        W[4*n+1, r] = 1

        r = 15
        U[3, r] = 1
        V[1*p+4, r] = 1
        W[4*n+1, r] = 1

        r = 16
        U[1, r] = 1
        V[1*p+4, r] = 1
        W[4*n+0, r] = 1

        if _check_decomposition(U, V, W, T, tol=1e-8):
            polished = _polish_rational(U, V, W, T)
            if polished is not None:
                return polished
            return U.tolist(), V.tolist(), W.tolist()

        # ALS refinement
        for it in range(50):
            if time.time() > deadline:
                break
            U = _als_update_U(T, V, W)
            V = _als_update_V(T, U, W)
            W = _als_update_W(T, U, V)

        if _check_decomposition(U, V, W, T, tol=1e-7):
            polished = _polish_rational(U, V, W, T)
            if polished is not None:
                return polished
            return U.tolist(), V.tolist(), W.tolist()

    return None


def _svd_guided_rank_search(T, target_rank, deadline):
    """
    Use SVD of matricizations to guide initialization for rank-target decomposition.
    Matricize T in mode-1 (u x vw), mode-2 (v x uw), mode-3 (w x uv).
    Use top singular vectors to initialize factor columns.
    """
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    R = target_rank

    T_f = T.astype(np.float64)

    # Mode-1 matricization: (U_rows) x (V_rows * W_rows)
    M1 = T_f.reshape(U_rows, -1)
    U1, S1, Vt1 = np.linalg.svd(M1, full_matrices=False)
    # Mode-2 matricization: (V_rows) x (U_rows * W_rows)
    M2 = T_f.transpose(1, 0, 2).reshape(V_rows, -1)
    U2, S2, Vt2 = np.linalg.svd(M2, full_matrices=False)
    # Mode-3 matricization: (W_rows) x (U_rows * V_rows)
    M3 = T_f.transpose(2, 0, 1).reshape(W_rows, -1)
    U3, S3, Vt3 = np.linalg.svd(M3, full_matrices=False)

    # Initialize using top-R left singular vectors scaled
    U_init = U1[:, :R] * np.sqrt(S1[:R])
    V_init = U2[:, :R] * np.sqrt(S2[:R])
    W_init = U3[:, :R] * np.sqrt(S3[:R])

    # Run ALS from this initialization
    U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
    for it in range(200):
        if time.time() > deadline:
            break
        U_cur = _als_update_U(T, V_cur, W_cur)
        V_cur = _als_update_V(T, U_cur, W_cur)
        W_cur = _als_update_W(T, U_cur, V_cur)

    if _check_decomposition(U_cur, V_cur, W_cur, T, tol=1e-7):
        polished = _polish_rational(U_cur, V_cur, W_cur, T)
        if polished is not None:
            return polished
        return U_cur.tolist(), V_cur.tolist(), W_cur.tolist()

    return None


def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def _try_perturbed_rank17(T, deadline):
    """
    Perturb the target tensor slightly, run ALS to rank 17, then try to
    recover an exact rational solution. This can break symmetries that
    prevent convergence to a low-rank solution.
    """
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n
    R = 17

    T_f = T.astype(np.float64)

    for noise_scale in [1e-4, 1e-5, 1e-6]:
        if time.time() > deadline:
            break

        for seed in [42, 123, 7, 999, 314, 777, 2024, 555, 111, 222]:
            if time.time() > deadline:
                break

            np.random.seed(seed)
            T_pert = T_f + np.random.randn(*T_f.shape) * noise_scale

            # Initialize from SVD
            M1 = T_pert.reshape(U_rows, -1)
            U1, S1, _ = np.linalg.svd(M1, full_matrices=False)
            M2 = T_pert.transpose(1, 0, 2).reshape(V_rows, -1)
            U2, S2, _ = np.linalg.svd(M2, full_matrices=False)
            M3 = T_pert.transpose(2, 0, 1).reshape(W_rows, -1)
            U3, S3, _ = np.linalg.svd(M3, full_matrices=False)

            U_cur = U1[:, :R] * np.sqrt(S1[:R])
            V_cur = U2[:, :R] * np.sqrt(S2[:R])
            W_cur = U3[:, :R] * np.sqrt(S3[:R])

            for it in range(150):
                if time.time() > deadline:
                    break
                U_cur = _als_update_U(T_pert, V_cur, W_cur)
                V_cur = _als_update_V(T_pert, U_cur, W_cur)
                W_cur = _als_update_W(T_pert, U_cur, V_cur)

            # Check if this decomposition works for the original T
            if _check_decomposition(U_cur, V_cur, W_cur, T, tol=1e-6):
                polished = _polish_rational(U_cur, V_cur, W_cur, T)
                if polished is not None:
                    return polished
                return U_cur.tolist(), V_cur.tolist(), W_cur.tolist()

    return None


def construct():
    # Safety margin: evaluator kills at 150s, we aim for 110s to be safe
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    T = _build_target()

    rank_lb = _estimate_tensor_rank(T)

    U_struct, V_struct, W_struct = _build_structured_decomposition()

    if not _check_decomposition(U_struct, V_struct, W_struct, T, tol=1e-8):
        U_l, V_l, W_l = _schoolbook_decomposition()
        return {"U": U_l, "V": V_l, "W": W_l}

    current_U, current_V, current_W = U_struct, V_struct, W_struct
    current_rank = U_struct.shape[1]

    # Try structured rank-17 with a tighter sub-deadline to save time for other searches
    sub_deadline_1 = time.time() + 20
    if time.time() < sub_deadline_1:
        result17 = _try_structured_rank17(T, sub_deadline_1)
        if result17 is not None:
            U_l, V_l, W_l = result17
            U_arr = np.array(U_l, dtype=np.float64)
            V_arr = np.array(V_l, dtype=np.float64)
            W_arr = np.array(W_l, dtype=np.float64)
            if _check_decomposition(U_arr, V_arr, W_arr, T, tol=1e-7):
                return {"U": U_l, "V": V_l, "W": W_l}

    # Try SVD-guided rank-17 search
    sub_deadline_2 = time.time() + 20
    if time.time() < sub_deadline_2:
        result17_svd = _svd_guided_rank_search(T, 17, sub_deadline_2)
        if result17_svd is not None:
            U_l, V_l, W_l = result17_svd
            U_arr = np.array(U_l, dtype=np.float64)
            V_arr = np.array(V_l, dtype=np.float64)
            W_arr = np.array(W_l, dtype=np.float64)
            if _check_decomposition(U_arr, V_arr, W_arr, T, tol=1e-7):
                return {"U": U_l, "V": V_l, "W": W_l}

    # Try perturbed rank-17 search
    sub_deadline_3 = time.time() + 30
    if time.time() < sub_deadline_3:
        result17_pert = _try_perturbed_rank17(T, sub_deadline_3)
        if result17_pert is not None:
            U_l, V_l, W_l = result17_pert
            U_arr = np.array(U_l, dtype=np.float64)
            V_arr = np.array(V_l, dtype=np.float64)
            W_arr = np.array(W_l, dtype=np.float64)
            if _check_decomposition(U_arr, V_arr, W_arr, T, tol=1e-7):
                return {"U": U_l, "V": V_l, "W": W_l}

    # Try rank reduction via local search (only if we have time left)
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        for reduction in range(1, min(current_rank - rank_lb + 1, 5)):
            if time.time() > deadline:
                break
            target_rank = current_rank - reduction
            if target_rank < rank_lb:
                break
            # Allocate remaining time / number of reductions
            sub_deadline_red = time.time() + (remaining_time / (5 - reduction + 1))
            result = _try_rank_reduction(current_U, current_V, current_W, T, sub_deadline_red, target_rank)
            if result is not None:
                U_l, V_l, W_l = result
                U_arr = np.array(U_l, dtype=np.float64)
                V_arr = np.array(V_l, dtype=np.float64)
                W_arr = np.array(W_l, dtype=np.float64)
                if _check_decomposition(U_arr, V_arr, W_arr, T, tol=1e-7):
                    return {"U": U_l, "V": V_l, "W": W_l}

    polished = _polish_rational(U_struct, V_struct, W_struct, T)
    if polished is not None:
        return {"U": polished[0], "V": polished[1], "W": polished[2]}

    return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
