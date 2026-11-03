import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 110.0


def _remaining():
    return _DEADLINE - time.time()


def _verify(U, V, W, n, m, p):
    """Verify T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                if abs(s - 1.0) > 1e-8:
                    return False
    return True


def _build_rank34(n, m, p):
    """
    Symmetry-reduced tensor network:
    - 2 shared Strassen 2x2x2 cores (one per B-column-pair), each feeding 2 A-blocks.
    - 20 schoolbook terms for the 5th column of B.
    Total rank: 2*7 + 20 = 34.
    """
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_shared_strassen(i0, j0, k0, col_start):
        """
        Shared Strassen core: C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        plus C[i0:i0+2, k0:k0+2] += A[i0:i0+2, j0+2:j0+4] * B[j0+2:j0+4, k0:k0+2]
        (same B sub-block, two A sub-blocks).
        """
        c = col_start
        # M1 = (A00+A11)(B00+B11)
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        U[i0 * m + (j0 + 2)][c] += 1
        U[(i0 + 1) * m + (j0 + 3)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M2 = (A10+A11)(B00)
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 2)][c] += 1
        U[(i0 + 1) * m + (j0 + 3)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        c += 1
        # M3 = (A00+A01)(B01-B11)
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        U[i0 * m + (j0 + 2)][c] += 1
        U[i0 * m + (j0 + 3)][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M4 = (A10+A11)(B10-B00)
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 2)][c] += 1
        U[(i0 + 1) * m + (j0 + 3)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        c += 1
        # M5 = (A00+A01)(B11)
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        U[i0 * m + (j0 + 2)][c] += 1
        U[i0 * m + (j0 + 3)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        c += 1
        # M6 = (A10-A00)(B00+B01)
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        U[(i0 + 1) * m + (j0 + 2)][c] += 1
        U[i0 * m + (j0 + 2)][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        c += 1
        # M7 = (A01-A11)(B10+B11)
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        U[i0 * m + (j0 + 3)][c] += 1
        U[(i0 + 1) * m + (j0 + 3)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        c += 1
        return c

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # Shared Strassen core 1: B[0:2, 0:2] feeding A[:, 0:2] and A[:, 2:4]
    col = add_shared_strassen(0, 0, 0, col)
    # Shared Strassen core 2: B[0:2, 2:4] feeding A[:, 0:2] and A[:, 2:4]
    col = add_shared_strassen(0, 0, 2, col)
    # Schoolbook for B[:, 4]
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    assert _verify(U, V, W, n, m, p), "Rank-34 decomposition failed verification"
    return U, V, W


def _try_reduce_columns(U, V, W, n, m, p):
    """Try to detect and remove a linearly dependent column from the rank-34 decomposition."""
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)

    # For each column r, check if it can be expressed as a linear combination of others.
    # The constraint is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1.
    # If column r is dependent, we need U_r = sum_{s != r} alpha_s * U_s (in the relevant sense).
    # This is a nonlinear constraint, so we check via: if we zero out column r, can we
    # adjust others to compensate? Simplification: check if the column is exactly zero
    # in a way that doesn't affect the tensor (unlikely for a minimal decomposition).

    # More practical: try removing each column and see if the remaining R-1 columns
    # still satisfy the tensor (they won't in general, but check for free reductions).
    for r in range(R):
        if _remaining() < 5.0:
            break
        U_test = [row[:] for row in U]
        V_test = [row[:] for row in V]
        W_test = [row[:] for row in W]
        # Zero out column r
        for i in range(n * m):
            U_test[i][r] = 0
        for j in range(m * p):
            V_test[j][r] = 0
        for k in range(p * n):
            W_test[k][r] = 0
        # Remove column r from all
        U_new = [row[:r] + row[r + 1:] for row in U_test]
        V_new = [row[:r] + row[r + 1:] for row in V_test]
        W_new = [row[:r] + row[r + 1:] for row in W_test]
        if _verify(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
    return None


def _local_search(U, V, W, n, m, p, max_trials=5000):
    """
    Local search: try to find a rank-(R-1) decomposition by perturbing the rank-R one.
    Strategy: for each pair of columns (r1, r2), try to merge them (set one to zero,
    scale the other) and check if the tensor is still satisfied. Also try small
    integer perturbations.
    """
    R = len(U[0])
    best_rank = R
    best_U, best_V, best_W = U, V, W

    np.random.seed(123)

    for target_rank in range(R - 1, max(20, R - 4)):
        if _remaining() < 10.0:
            break

        for trial in range(max_trials):
            if _remaining() < 5.0:
                break

            # Strategy: take the rank-R solution, zero out one column, and try to
            # compensate by adjusting another column.
            r_zero = np.random.randint(0, R)
            r_adj = np.random.randint(0, R)
            if r_zero == r_adj:
                continue

            U_try = [row[:] for row in U]
            V_try = [row[:] for row in V]
            W_try = [row[:] for row in W]

            # Zero out column r_zero
            for i in range(n * m):
                U_try[i][r_zero] = 0
            for j in range(m * p):
                V_try[j][r_zero] = 0
            for k in range(p * n):
                W_try[k][r_zero] = 0

            # Try scaling column r_adj by small factors
            scale = np.random.choice([1, 2, -1, -2, 0.5])
            for i in range(n * m):
                U_try[i][r_adj] = U[i][r_adj] * scale
            for j in range(m * p):
                V_try[j][r_adj] = V[j][r_adj] * scale
            for k in range(p * n):
                W_try[k][r_adj] = W[k][r_adj] * scale

            # Remove the zeroed column
            U_new = [row[:r_zero] + row[r_zero + 1:] for row in U_try]
            V_new = [row[:r_zero] + row[r_zero + 1:] for row in V_try]
            W_new = [row[:r_zero] + row[r_zero + 1:] for row in W_try]

            if _verify(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new

    return best_U, best_V, best_W


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.

    Uses symmetry-reduced tensor network: 2 shared Strassen cores (rank 14)
    + 20 schoolbook terms for the 5th column = rank 34.
    Then attempts further rank reduction via local search.
    """
    n, m, p = 2, 4, 5

    # Build the rank-34 symmetry-reduced decomposition
    U, V, W = _build_rank34(n, m, p)
    best_rank = 34
    best_U, best_V, best_W = U, V, W

    # Try to reduce via column dependency detection
    if _remaining() > 15.0:
        reduced = _try_reduce_columns(U, V, W, n, m, p)
        if reduced is not None:
            best_U, best_V, best_W = reduced
            best_rank = len(best_U[0])

    # Try local search for further reduction
    if _remaining() > 20.0:
        U_ls, V_ls, W_ls = _local_search(best_U, best_V, best_W, n, m, p, max_trials=3000)
        if len(U_ls[0]) < best_rank:
            best_U, best_V, best_W = U_ls, V_ls, W_ls
            best_rank = len(best_U[0])

    # Final verification
    assert _verify(best_U, best_V, best_W, n, m, p), "Final decomposition failed verification"

    return {"U": best_U, "V": best_V, "W": best_W}
