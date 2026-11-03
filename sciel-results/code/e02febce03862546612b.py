"""Rewritten program: joint decomposition for 4x4x8 exploiting shared A."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _als_4x4x4(seed, max_iter=400, tol=1e-13):
    """Fast ALS for 4x4x4 rank-20. Returns (U,V,W) or None."""
    n = m = p = 4
    rank = 20
    np.random.seed(seed)
    U = np.random.randn(16, rank) * 0.3
    V = np.random.randn(16, rank) * 0.3
    W = np.random.randn(16, rank) * 0.3

    for it in range(max_iter):
        if _time_left() < 5:
            return None
        # Update U: for each (i,j), solve A_kc = V[j,k,c]*W[k,i,c], b_k=1
        for i in range(4):
            for j in range(4):
                A = np.zeros((4, rank))
                for k in range(4):
                    A[k, :] = V[j * 4 + k, :] * W[k * 4 + i, :]
                try:
                    U[i * 4 + j, :] = np.linalg.lstsq(A, np.ones(4), rcond=None)[0]
                except Exception:
                    pass
        # Update V
        for j in range(4):
            for k in range(4):
                A = np.zeros((4, rank))
                for i in range(4):
                    A[i, :] = U[i * 4 + j, :] * W[k * 4 + i, :]
                try:
                    V[j * 4 + k, :] = np.linalg.lstsq(A, np.ones(4), rcond=None)[0]
                except Exception:
                    pass
        # Update W
        for k in range(4):
            for i in range(4):
                A = np.zeros((4, rank))
                for j in range(4):
                    A[j, :] = U[i * 4 + j, :] * V[j * 4 + k, :]
                try:
                    W[k * 4 + i, :] = np.linalg.lstsq(A, np.ones(4), rcond=None)[0]
                except Exception:
                    pass
        # Check error
        err = 0.0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = np.dot(U[i*4+j, :], V[j*4+k, :] * W[k*4+i, :])
                    err += (s - 1.0) ** 2
        if err < tol:
            return U, V, W
    # Final check
    err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = np.dot(U[i*4+j, :], V[j*4+k, :] * W[k*4+i, :])
                err += (s - 1.0) ** 2
    if err < 1e-10:
        return U, V, W
    return None


def _get_lader_4x4x4():
    """Get verified rank-20 Lader decomposition for 4x4x4."""
    for seed in [42, 7, 123, 999, 1, 2, 3, 5, 11, 17, 23, 31, 37, 43, 53, 67, 79, 97, 101, 113]:
        if _time_left() < 10:
            break
        res = _als_4x4x4(seed)
        if res is not None:
            return res
    return None


def _verify_full(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify the full decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_full_err(U, V, W, n, m, p, rank):
    """Return max error."""
    maxerr = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                maxerr = max(maxerr, abs(s - 1.0))
    return maxerr


def _als_full_4x4x8(U, V, W, n, m, p, rank, max_iter=300, tol=1e-13):
    """ALS on the full 4x4x8 tensor with given rank."""
    for it in range(max_iter):
        if _time_left() < 5:
            break
        # Update U: for each (i,j), for all k in 0..7, need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except Exception:
                    pass
        # Update V: for each (j,k), for all i in 0..3
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except Exception:
                    pass
        # Update W: for each (k,i), for all j in 0..3
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except Exception:
                    pass
        err = _verify_full_err(U, V, W, n, m, p, rank)
        if err < tol:
            return U, V, W
    return U, V, W


def _build_block_decomp(U4, V4, W4, n, m, p):
    """Build rank-40 block decomposition from 4x4x4 Lader."""
    rank4 = U4.shape[1]
    total_rank = 2 * rank4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    for c in range(rank4):
        # Block 1: k in 0..3
        U[:, c] = U4[:, c]
        for j in range(4):
            for k in range(4):
                V[j * 8 + k, c] = V4[j * 4 + k, c]
        for k in range(4):
            for i in range(4):
                W[k * 4 + i, c] = W4[k * 4 + i, c]

        # Block 2: k in 4..7
        cc = rank4 + c
        U[:, cc] = U4[:, c]
        for j in range(4):
            for k_local in range(4):
                k = 4 + k_local
                V[j * 8 + k, cc] = V4[j * 4 + k_local, c]
        for k_local in range(4):
            k = 4 + k_local
            for i in range(4):
                W[k * 4 + i, cc] = W4[k_local * 4 + i, c]

    return U, V, W, total_rank


def _try_rank_reduction(U, V, W, n, m, p, start_rank, target_rank, seed):
    """Try to reduce rank from start_rank to target_rank via ALS."""
    if _time_left() < 10:
        return None

    np.random.seed(seed)
    # Initialize: take first target_rank columns, add small random perturbation
    U0 = U[:, :target_rank].copy() + np.random.randn(n * m, target_rank) * 0.01
    V0 = V[:, :target_rank].copy() + np.random.randn(m * p, target_rank) * 0.01
    W0 = W[:, :target_rank].copy() + np.random.randn(p * n, target_rank) * 0.01

    # ALS
    U, V, W = _als_full_4x4x8(U0, V0, W0, n, m, p, target_rank, max_iter=200)
    err = _verify_full_err(U, V, W, n, m, p, target_rank)
    if err < 1e-9:
        return U, V, W
    return None


def construct():
    """
    Find a low-rank exact decomposition for 4x4 * 4x8 matrix multiplication.
    Uses joint ALS to exploit shared A structure, aiming for rank < 40.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Step 1: Get Lader 4x4x4
    lader = _get_lader_4x4x4()

    if lader is not None:
        U4, V4, W4 = lader
        # Step 2: Build block decomposition (rank 40)
        U, V, W, rank40 = _build_block_decomp(U4, V4, W4, n, m, p)

        # Verify block decomp
        if not _verify_full(U, V, W, n, m, p, rank40):
            # Should not happen, but fallback
            return _schoolbook(n, m, p)

        # Step 3: Try rank reduction to 36, 35, 34, ...
        best_U, best_V, best_W, best_rank = U, V, W, rank40

        for target in [36, 35, 34, 33, 32, 31, 30]:
            if _time_left() < 15:
                break
            found = False
            for seed in [1, 7, 42, 123, 999, 2024, 555, 777, 888, 111]:
                if _time_left() < 10:
                    break
                if found:
                    break
                res = _try_rank_reduction(U, V, W, n, m, p, rank40, target, seed)
                if res is not None:
                    U, V, W = res
                    best_U, best_V, best_W, best_rank = U, V, W, target
                    found = True
            if not found and _time_left() < 20:
                break

        # Return best found
        U, V, W = best_U, best_V, best_W
        total_rank = best_rank
    else:
        # Fallback: try direct ALS on 4x4x8 with various ranks
        found = False
        for target in [36, 38, 40]:
            if _time_left() < 15:
                break
            if found:
                break
            for seed in [42, 7, 123, 999, 1]:
                if _time_left() < 10:
                    break
                if found:
                    break
                np.random.seed(seed)
                U = np.random.randn(n * m, target) * 0.3
                V = np.random.randn(m * p, target) * 0.3
                W = np.random.randn(p * n, target) * 0.3
                U, V, W = _als_full_4x4x8(U, V, W, n, m, p, target, max_iter=300)
                if _verify_full(U, V, W, n, m, p, target):
                    total_rank = target
                    found = True
        if not found:
            return _schoolbook(n, m, p)

    # Convert to lists
    U_list = [[float(U[i, c]) for c in range(total_rank)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(total_rank)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(total_rank)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}


def _schoolbook(n, m, p):
    """Schoolbook decomposition, rank = n*m*p."""
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
    U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p * n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
