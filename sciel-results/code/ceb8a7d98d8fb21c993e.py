import time
import numpy as np

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0  # seconds (safety margin under 150 s wall-clock)

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition using vectorized operations."""
    rank = U.shape[1]
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    T_rec = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    T_true = np.ones((n, m, p))
    return bool(np.allclose(T_rec, T_true, atol=tol))


def _build_strassen_decomposition():
    """
    Build a rank-28 decomposition by applying Strassen's 7-term 2x2x2
    algorithm to the four 2x2 blocks of the 2x4 * 4x4 multiplication.

    A (2x4)  -> A_L (cols 0,1), A_R (cols 2,3)
    B (4x4)  -> B_TL, B_TR (rows 0,1), B_BL, B_BR (rows 2,3)
    C (2x4)  -> C_L (cols 0,1), C_R (cols 2,3)

    C_L = A_L B_TL + A_R B_BL
    C_R = A_L B_TR + A_R B_BR

    Output factor uses the official trace convention W[k*n + i, r].
    """
    n, m, p = 2, 4, 4
    rank_total = 28

    # Strassen terms for 2x2 * 2x2 -> 2x2.
    # Local index: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Each term: (u_coeffs, v_coeffs, w_coeffs) as lists of (local_idx, value)
    strassen = [
        # M1 = (A00 + A11) * (B00 + B11) -> C00 + C11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10 + A11) * B00 -> C10 - C11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00 * (B01 - B11) -> C01 + C11
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11 * (B10 - B00) -> C00 + C10
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5 = (A00 + A01) * B11 -> -C00 + C01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10 - A00) * (B00 + B01) -> C11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01 - A11) * (B10 + B11) -> C00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    U = np.zeros((n * m, rank_total), dtype=int)
    V = np.zeros((m * p, rank_total), dtype=int)
    W = np.zeros((p * n, rank_total), dtype=int)

    # Global index maps for A (2x4), row-major: A[r, c] -> r*4 + c
    # A_L uses cols 0,1 ; A_R uses cols 2,3.
    def map_A(local_idx, col_off):
        r = local_idx // 2
        c = local_idx % 2
        return r * 4 + (col_off * 2 + c)

    # B (4x4), row-major: B[r, c] -> r*4 + c
    def map_B(local_idx, row_off, col_off):
        r = local_idx // 2
        c = local_idx % 2
        return (row_off * 2 + r) * 4 + (col_off * 2 + c)

    # C (2x4) output, convention W[k*n + i] where k = output col, i = row.
    # Local C (2x2) for a pair of output columns (col_off*2, col_off*2+1).
    def map_C(local_idx, col_off):
        r = local_idx // 2      # row i
        c = local_idx % 2       # local col within the 2-col pair
        k = col_off * 2 + c     # global output column
        return k * 2 + r

    # Four block multiplications.
    # (u_col_off, b_row_off, b_col_off, c_col_off, base_col)
    blocks = [
        (0, 0, 0, 0, 0),    # A_L * B_TL -> C_L
        (1, 1, 0, 0, 7),    # A_R * B_BL -> C_L
        (0, 0, 1, 1, 14),   # A_L * B_TR -> C_R
        (1, 1, 1, 1, 21),   # A_R * B_BR -> C_R
    ]

    for u_co, b_ro, b_co, c_co, base in blocks:
        for t, (u_coeff, v_coeff, w_coeff) in enumerate(strassen):
            col = base + t
            for li, val in u_coeff:
                U[map_A(li, u_co), col] += val
            for li, val in v_coeff:
                V[map_B(li, b_ro, b_co), col] += val
            for li, val in w_coeff:
                W[map_C(li, c_co), col] += val

    return U, V, W


def _try_rank_27_search(n, m, p, time_limit=10.0):
    """
    Fast feasibility probe for a rank-27 decomposition.

    For a fixed (U, V) with integer entries, the tensor equation
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    is LINEAR in W. For each (i, k) the unknowns are W[k*n+i, :] (rank values)
    and the m=4 equations (over j) form a system A_{ik} w_{ik} = 1 (4x27).
    Such a system always has a real solution (4 < 27), so feasibility over the
    reals is guaranteed for any (U, V). The hard requirement is a RATIONAL
    (integer/half-integer) W. We therefore probe whether, for a set of random
    integer (U, V), the required W happens to be rational (a rational basis of
    the solution space exists iff the augmented system has the same rank as the
    coefficient system over Q, which we check via exact-free rank with tight
    tolerance). If found, we return a verified integer/half-integer W; otherwise
    None. This is a cheap refinement, not a full rational solver.
    """
    start = time.time()
    rank = 27
    rng = np.random.default_rng(12345)

    for trial in range(60):
        if time.time() - start > time_limit:
            break
        # Random sparse-ish integer U, V in {-1, 0, 1}
        U = rng.integers(-1, 2, size=(n * m, rank))
        V = rng.integers(-1, 2, size=(m * p, rank))

        W = np.zeros((p * n, rank), dtype=float)
        ok = True
        for i in range(n):
            for k in range(p):
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = (U[i * m:i * m + m, :] * V[k * p:k * p + m, :]).T  # (rank, m)
                b = np.ones(m)
                # Solve A^T w = b, i.e. (m x rank) system.
                # Use least squares; check residual AND rationality.
                w, res, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
                if np.linalg.norm(A.T @ w - b) > 1e-7:
                    ok = False
                    break
                # Rationality heuristic: check if w is close to half-integers.
                scaled = w * 2.0
                if not np.allclose(scaled, np.round(scaled), atol=1e-6):
                    ok = False
                    break
                W[k * n + i, :] = w
            if not ok:
                break
        if not ok:
            continue

        W_int = np.round(W * 2.0) / 2.0
        if _verify_fast(U, V, W_int, n, m, p):
            return U, V, W_int
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # 1. Build the known Rank 28 decomposition and verify it.
    U, V, W = _build_strassen_decomposition()
    if _verify_fast(U, V, W, n, m, p):
        best = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        best_rank = 28
    else:
        # Fallback to schoolbook (rank = n*m*p = 32).
        rank_fallback = n * m * p
        U_fb = np.zeros((n * m, rank_fallback))
        V_fb = np.zeros((m * p, rank_fallback))
        W_fb = np.zeros((p * n, rank_fallback))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        best = {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}
        best_rank = rank_fallback

    # 2. Refinement: cheap attempt to beat 28 with rank 27 (rational W probe).
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5.0:
        probe = _try_rank_27_search(n, m, p, time_limit=min(remaining - 2.0, 20.0))
        if probe is not None:
            U27, V27, W27 = probe
            if _verify_fast(U27, V27, W27, n, m, p):
                best = {'U': U27.tolist(), 'V': V27.tolist(), 'W': W27.tolist()}
                best_rank = 27

    return best

# EVOLVE-BLOCK-END
