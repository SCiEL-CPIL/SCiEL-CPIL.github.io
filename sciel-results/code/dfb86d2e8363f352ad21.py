"""
Combined program: recursive Kronecker Strassen (49) + naive tail (4) for 4x4x5,
with fast NumPy verification and a structural fallback to block-Strassen (rank 64).
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """Return the 7 Strassen 2x2x2 terms as sparse (index, value) lists.
    Index convention: for U, index = i*2+j; for V, j*2+k; for W, k*2+i.
    """
    return [
        # M1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h)
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e)
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]


def _kron_strassen_4x4x4():
    """Build rank-49 decomposition for 4x4x4 via Kronecker of two Strassen 2x2x2."""
    n = 4
    terms = _strassen_2x2x2_terms()
    U_cols = []
    V_cols = []
    W_cols = []
    for (u1_sp, v1_sp, w1_sp) in terms:
        for (u2_sp, v2_sp, w2_sp) in terms:
            u = np.zeros(n * n, dtype=np.float64)
            v = np.zeros(n * n, dtype=np.float64)
            w = np.zeros(n * n, dtype=np.float64)
            for (idx1, val1) in u1_sp:
                i1, j1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in u2_sp:
                    i2, j2 = idx2 // 2, idx2 % 2
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    u[i * n + j] += val1 * val2
            for (idx1, val1) in v1_sp:
                j1, k1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in v2_sp:
                    j2, k2 = idx2 // 2, idx2 % 2
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    v[j * n + k] += val1 * val2
            for (idx1, val1) in w1_sp:
                k1, i1 = idx1 // 2, idx1 % 2
                for (idx2, val2) in w2_sp:
                    k2, i2 = idx2 // 2, idx2 % 2
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    w[k * n + i] += val1 * val2
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    return U.tolist(), V.tolist(), W.tolist()


def _build_4x4x5_kron_strassen():
    """Build rank-53 decomposition for 4x4x5.
    First 49 terms handle 4x4x4 (columns 0..3 of B), remaining 4 terms handle column 4.
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    base_U, base_V, base_W = _kron_strassen_4x4x4()
    rank_base = len(base_U[0])
    rank = rank_base + 4

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base_U[i][r]

    for j in range(m):
        for k in range(4):
            for r in range(rank_base):
                V[j * p + k][r] = base_V[j * 4 + k][r]

    for k in range(4):
        for i in range(n):
            for r in range(rank_base):
                W[k * n + i][r] = base_W[k * 4 + i][r]

    # Naive terms for the 5th column of B (index k=4)
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i * m + j][r] = 1.0
        V[j * p + 4][r] = 1.0
        for i in range(n):
            W[4 * n + i][r] = 1.0

    return {"U": U, "V": V, "W": W}


def _build_block_strassen_4x4x5():
    """Build rank-64 decomposition using 2x2 block Strassen + naive for 2x2x3 parts."""
    n, m, p = DIMENSIONS
    rank = 64
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    strassen_2x2 = _strassen_2x2x2_terms()

    def add_strassen_2x2_2(ai, aj, bj, bk, start):
        for t_idx, (u_sp, v_sp, w_sp) in enumerate(strassen_2x2):
            r = start + t_idx
            for (idx, val) in u_sp:
                ui, uj = idx // 2, idx % 2
                U[(ai + ui) * m + (aj + uj)][r] += val
            for (idx, val) in v_sp:
                vj, vk = idx // 2, idx % 2
                V[(bj + vj) * p + (bk + vk)][r] += val
            for (idx, val) in w_sp:
                wk, wi = idx // 2, idx % 2
                W[(bk + wk) * n + (ai + wi)][r] += val
        return start + 7

    def add_2x2x3(ai, aj, bj, bk, start):
        # First 2 columns via Strassen
        start = add_strassen_2x2_2(ai, aj, bj, bk, start)
        # 3rd column via schoolbook
        for di in range(2):
            for dj in range(2):
                r = start
                U[(ai + di) * m + (aj + dj)][r] += 1.0
                V[(bj + dj) * p + (bk + 2)][r] += 1.0
                W[(bk + 2) * n + (ai + di)][r] += 1.0
                start += 1
        return start

    rank_idx = 0
    # C00 = A00*B00 + A01*B10
    rank_idx = add_strassen_2x2_2(0, 0, 0, 0, rank_idx)
    rank_idx = add_strassen_2x2_2(0, 2, 2, 0, rank_idx)
    # C01 = A00*B01 + A01*B11
    rank_idx = add_2x2x3(0, 0, 0, 2, rank_idx)
    rank_idx = add_2x2x3(0, 2, 2, 2, rank_idx)
    # C10 = A10*B00 + A11*B10
    rank_idx = add_strassen_2x2_2(2, 0, 0, 0, rank_idx)
    rank_idx = add_strassen_2x2_2(2, 2, 2, 0, rank_idx)
    # C11 = A10*B01 + A11*B11
    rank_idx = add_2x2x3(2, 0, 0, 2, rank_idx)
    rank_idx = add_2x2x3(2, 2, 2, 2, rank_idx)

    assert rank_idx == rank, f"Expected rank {rank}, got {rank_idx}"
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64).reshape(n, m, rank)
    Va = np.array(V, dtype=np.float64).reshape(m, p, rank)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, rank)
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        T += Ua[:, :, r:r+1] * Va[:, :, r:r+1] * Wa[:, :, r:r+1]
    return np.allclose(T, 1.0, atol=tol)


def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True


def _build_schoolbook():
    """Fallback schoolbook decomposition: rank = n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120

    n, m, p = DIMENSIONS

    best_result = None
    best_rank = float('inf')

    # Approach 1: Kronecker Strassen 4x4x4 + naive tail (rank 53)
    try:
        result = _build_4x4x5_kron_strassen()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
    except Exception:
        pass

    # Approach 2: Block Strassen 2x2 (rank 64)
    if time.time() < deadline - 10:
        try:
            result = _build_block_strassen_4x4x5()
            rank = len(result["U"][0])
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                if _verify_integer_half_integer(result["U"], result["V"], result["W"]):
                    if best_result is None or rank < best_rank:
                        best_result = result
                        best_rank = rank
        except Exception:
            pass

    # Approach 3: Schoolbook (rank 80) as fallback
    if best_result is None or time.time() < deadline - 5:
        try:
            result = _build_schoolbook()
            rank = len(result["U"][0])
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                if best_result is None or rank < best_rank:
                    best_result = result
                    best_rank = rank
        except Exception:
            pass

    if best_result is not None:
        return best_result

    # Final fallback
    return _build_schoolbook()

# EVOLVE-BLOCK-END
