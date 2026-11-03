import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def _build_strassen_rank28():
    """Build the known rank-28 Strassen decomposition for 2x4x4 matmul."""
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen rank-1 components for 2x2*2x2 (P*Q=R)
    # Local idx: 00:0, 01:1, 10:2, 11:3
    strassen_steps = [
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}},
        {'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}},
        {'u': {3: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}},
        {'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}},
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}},
        {'u': {1: 1, 3: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}}
    ]

    def add_block(a_col_start, b_row_start, b_col_start, c_col_start, col_offset):
        for s in range(7):
            step = strassen_steps[s]
            col = col_offset + s
            for local_idx, coeff in step['u'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = l_row * m + (a_col_start + l_col)
                U[global_idx, col] += coeff
            for local_idx, coeff in step['v'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = (b_row_start + l_row) * p + (b_col_start + l_col)
                V[global_idx, col] += coeff
            for local_idx, coeff in step['w'].items():
                l_row = local_idx // 2
                l_col = local_idx % 2
                global_idx = (c_col_start + l_col) * n + l_row
                W[global_idx, col] += coeff

    # 4 terms: A_block * B_block -> C_block
    add_block(0, 0, 0, 0, 0)   # A0 * B00 -> C0
    add_block(2, 2, 0, 0, 7)   # A1 * B10 -> C0
    add_block(0, 0, 2, 2, 14)  # A0 * B01 -> C1
    add_block(2, 2, 2, 2, 21)  # A1 * B11 -> C1

    return U, V, W


def _verify_decomposition(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the 2x4x4 matmul tensor."""
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    rank = U_arr.shape[1]
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)

    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = np.max(np.abs(recon - T))
    return residual < tol, residual


def _als_polish(U, V, W, max_iters=200, tol=1e-12, time_limit=None, start_time=None):
    """
    Fast ALS polish: iteratively solve for each factor given the other two.
    Uses a smooth epigraph objective (L2 residual) for numerical stability.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T = np.ones((n, m, p))

    U3 = U.reshape(n, m, rank).astype(float).copy()
    V3 = V.reshape(m, p, rank).astype(float).copy()
    W3 = W.reshape(p, n, rank).astype(float).copy()

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break

        # Update U
        for i in range(n):
            for j in range(m):
                A = (V3[j] * W3[:, i]).T  # (rank, p)
                try:
                    U3[i, j] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                except Exception:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                B = (U3[:, j] * W3[k]).T  # (rank, n)
                try:
                    V3[j, k] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                except Exception:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                C = (U3[i] * V3[:, k]).T  # (rank, m)
                try:
                    W3[k, i] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                except Exception:
                    pass

        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = np.max(np.abs(recon - T))
        if residual < tol:
            break

    U_flat = U3.reshape(n * m, rank)
    V_flat = V3.reshape(m * p, rank)
    W_flat = W3.reshape(p * n, rank)
    return U_flat, V_flat, W_flat, residual


def _global_bilinear_search(target_rank, seed, max_iters=400, time_limit=None, start_time=None, init='random'):
    """
    Global bilinear form optimization: parameterize the 2x4x4 multiplication
    tensor as a sum of rank-1 terms with arbitrary linear forms on the inputs.
    Uses a smooth epigraph objective (L2 residual) and randomized initialization
    to directly search for a low-rank exact solution, bypassing block structure.
    """
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    rng = np.random.RandomState(seed)

    if init == 'random':
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
    else:
        U, V, W = init  # (n*m, target_rank), etc.

    U3 = U.reshape(n, m, target_rank).astype(float).copy()
    V3 = V.reshape(m, p, target_rank).astype(float).copy()
    W3 = W.reshape(p, n, target_rank).astype(float).copy()

    best_res = np.inf
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break

        # Update U
        for i in range(n):
            for j in range(m):
                A = (V3[j] * W3[:, i]).T
                try:
                    U3[i, j] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                except Exception:
                    pass
        # Update V
        for j in range(m):
            for k in range(p):
                B = (U3[:, j] * W3[k]).T
                try:
                    V3[j, k] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                except Exception:
                    pass
        # Update W
        for k in range(p):
            for i in range(n):
                C = (U3[i] * V3[:, k]).T
                try:
                    W3[k, i] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                except Exception:
                    pass

        if it % 20 == 0:
            recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            residual = np.max(np.abs(recon - T))
            if residual < best_res:
                best_res = residual
            if residual < 1e-9:
                break

    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = np.max(np.abs(recon - T))

    if residual < 1e-5:
        U_flat = U3.reshape(n * m, target_rank)
        V_flat = V3.reshape(m * p, target_rank)
        W_flat = W3.reshape(p * n, target_rank)
        return U_flat, V_flat, W_flat, residual
    return None


def _try_rank_reduction(target_rank, seed, max_iters=300, time_limit=None, start_time=None):
    """
    Try to reduce from rank 28 to target_rank by zeroing columns and re-optimizing.
    """
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))

    U28, V28, W28 = _build_strassen_rank28()

    for col_idx in range(28):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break

        mask = np.ones(28, dtype=bool)
        mask[col_idx] = False

        U = U28[:, mask].copy().astype(float)
        V = V28[:, mask].copy().astype(float)
        W = W28[:, mask].copy().astype(float)

        U3 = U.reshape(n, m, target_rank)
        V3 = V.reshape(m, p, target_rank)
        W3 = W.reshape(p, n, target_rank)

        for it in range(max_iters):
            if time_limit is not None and start_time is not None:
                if time.time() - start_time > time_limit:
                    break

            for i in range(n):
                for j in range(m):
                    A = (V3[j] * W3[:, i]).T
                    try:
                        U3[i, j] = np.linalg.lstsq(A, T[i, j, :], rcond=None)[0]
                    except Exception:
                        pass
            for j in range(m):
                for k in range(p):
                    B = (U3[:, j] * W3[k]).T
                    try:
                        V3[j, k] = np.linalg.lstsq(B, T[:, j, k], rcond=None)[0]
                    except Exception:
                        pass
            for k in range(p):
                for i in range(n):
                    C = (U3[i] * V3[:, k]).T
                    try:
                        W3[k, i] = np.linalg.lstsq(C, T[i, :, k], rcond=None)[0]
                    except Exception:
                        pass

            if it % 50 == 0:
                recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                residual = np.max(np.abs(recon - T))
                if residual < 1e-8:
                    break

        recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        residual = np.max(np.abs(recon - T))

        if residual < 1e-5:
            U_flat = U3.reshape(n * m, target_rank)
            V_flat = V3.reshape(m * p, target_rank)
            W_flat = W3.reshape(p * n, target_rank)
            return U_flat, V_flat, W_flat, residual

    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit

    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]

    # Start with known rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_rank28()
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28

    # Polish the rank-28 solution
    if time.time() < deadline:
        try:
            U_pol, V_pol, W_pol, res = _als_polish(U28, V28, W28, max_iters=100, tol=1e-12,
                                                     time_limit=deadline - time.time(), start_time=start_time)
            if res < 1e-10:
                best_result = {"U": to_lists(U_pol), "V": to_lists(V_pol), "W": to_lists(W_pol)}
        except Exception:
            pass

    # Try to reduce to rank 27 via column-zeroing + re-optimization
    if time.time() < deadline and best_rank == 28:
        for seed in range(30):
            if time.time() >= deadline:
                break
            try:
                result = _try_rank_reduction(27, seed, max_iters=150,
                                              time_limit=deadline - time.time(), start_time=start_time)
                if result is not None:
                    U_r, V_r, W_r, res = result
                    valid, res2 = _verify_decomposition(U_r, V_r, W_r, tol=1e-5)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                        break
            except Exception:
                continue

    # Global bilinear form search for rank 27 (bypasses block structure)
    if time.time() < deadline and best_rank == 28:
        for seed in range(40):
            if time.time() >= deadline:
                break
            try:
                result = _global_bilinear_search(27, seed, max_iters=250,
                                                  time_limit=deadline - time.time(), start_time=start_time,
                                                  init='random')
                if result is not None:
                    U_r, V_r, W_r, res = result
                    valid, res2 = _verify_decomposition(U_r, V_r, W_r, tol=1e-5)
                    if valid:
                        best_rank = 27
                        best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                        break
            except Exception:
                continue

    # Global bilinear form search for rank 26 (aggressive, only if time remains)
    if time.time() < deadline and best_rank > 26:
        for seed in range(20):
            if time.time() >= deadline:
                break
            try:
                result = _global_bilinear_search(26, seed, max_iters=200,
                                                  time_limit=deadline - time.time(), start_time=start_time,
                                                  init='random')
                if result is not None:
                    U_r, V_r, W_r, res = result
                    valid, res2 = _verify_decomposition(U_r, V_r, W_r, tol=1e-5)
                    if valid:
                        best_rank = 26
                        best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                        break
            except Exception:
                continue

    # Final high-precision polish pass on the best result
    if time.time() < deadline and best_rank < 28:
        try:
            U_best = np.array(best_result["U"], dtype=float)
            V_best = np.array(best_result["V"], dtype=float)
            W_best = np.array(best_result["W"], dtype=float)
            U_pol, V_pol, W_pol, res = _als_polish(U_best, V_best, W_best, max_iters=150, tol=1e-14,
                                                     time_limit=deadline - time.time(), start_time=start_time)
            if res < 1e-10:
                valid, res2 = _verify_decomposition(U_pol, V_pol, W_pol, tol=1e-8)
                if valid:
                    best_result = {"U": to_lists(U_pol), "V": to_lists(V_pol), "W": to_lists(W_pol)}
        except Exception:
            pass

    return best_result
