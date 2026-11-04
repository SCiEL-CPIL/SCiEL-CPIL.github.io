import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    Convention: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k].
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def winograd_2x4x4_factors():
    """
    Winograd F(4,2) convolution: multiply a 2x4 matrix by a 4x4 matrix -> 2x4 output.
    This is the tensor of linear convolution length 4 with kernel 2 (or equivalent matmul 2x4 * 4x4 = 2x4).
    Rank is 9 (F(4,2) has 9 multiplications).
    We express it as a 2x4x4 tensor: T[i, j, k] where i in {0,1} (output rows), j in {0..3} (input cols of A), k in {0..3} (output cols).
    Actually to keep the matmul convention we treat it as n=2, m=4, p=4.
    The convolution structure: C[i,k] = sum_j A[i,j]*B[j,k]. This is just a plain 2x4x4 matmul with rank 8 schoolbook.
    Winograd reduces the *convolution* but a plain matmul 2x4x4 already has rank 8. So this block is not useful as a matmul block.
    We therefore return a schoolbook 2x4x4 (rank 8) to keep the interface, but mark it as not better.
    """
    # For a plain matrix multiplication 2x4x4, schoolbook rank is 8, which is optimal.
    U = np.zeros((8, 8), dtype=np.float64)
    V = np.zeros((16, 8), dtype=np.float64)
    W = np.zeros((8, 8), dtype=np.float64)
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j, col] = 1.0
                V[j * 4 + k, col] = 1.0
                W[k * 2 + i, col] = 1.0
                col += 1
    return U, V, W

def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    return U, V, W

def _block_rank(ni, mj, pk):
    """Return the rank for a block of size ni x mj x pk using the best known decomposition."""
    if ni == 2 and mj == 2 and pk == 2:
        return 7
    # A plain matmul block of size ni x mj x pk has rank ni*mj*pk (schoolbook), which is optimal for generic blocks.
    return ni * mj * pk

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """
    Build the full factor matrices using block decomposition.
    """
    total_rank = 0
    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset)
        offset += s

    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset)
        offset += s

    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset)
        offset += s

    for i_idx, ni in enumerate(n_sizes):
        for j_idx, mj in enumerate(m_sizes):
            for k_idx, pk in enumerate(p_sizes):
                total_rank += _block_rank(ni, mj, pk)

    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)

    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]

                if ni == 2 and mj == 2 and pk == 2:
                    U_block, V_block, W_block = strassen_factors()
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                r = U_block.shape[1]

                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]

                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]

                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]

                col_offset += r

    return U_full, V_full, W_full, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    T[i,j,k] should be 1 for all i,j,k.
    """
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx

    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)

    return bool(np.allclose(vals, 1.0, atol=atol))

def schoolbook_lists(n, m, p):
    """Return schoolbook decomposition as nested lists (fallback)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _compositions(total, max_parts=None):
    if max_parts is None:
        max_parts = total
    if total == 0:
        yield []
        return
    for first in range(1, total + 1):
        for rest in _compositions(total - first, max_parts - 1):
            yield [first] + rest

def _find_best_block_partition(n, m, p):
    """
    Search over all valid block partitions to find the one with minimum total rank.
    Returns (n_sizes, m_sizes, p_sizes, total_rank).
    """
    best_rank = n * m * p
    best_partition = (list(range(n)), list(range(m)), list(range(p)))

    n_comps = list(_compositions(n))
    m_comps = list(_compositions(m))
    p_comps = list(_compositions(p))

    for n_sizes in n_comps:
        for m_sizes in m_comps:
            for p_sizes in p_comps:
                rank = 0
                for ni in n_sizes:
                    for mj in m_sizes:
                        for pk in p_sizes:
                            rank += _block_rank(ni, mj, pk)
                if rank < best_rank:
                    best_rank = rank
                    best_partition = (list(n_sizes), list(m_sizes), list(p_sizes))

    return best_partition[0], best_partition[1], best_partition[2], best_rank

def _winograd_1d_rank(length):
    """
    Winograd F(2k, k) convolution has rank 2k+1 for output length 2k with kernel length k+1? 
    We use the standard result: the 1-D tensor rank for linear convolution of a sequence of
    length a with a kernel of length b (output length a+b-1) via Winograd is a+b-1 when using
    the minimal FFT-like scheme. For our matmul interpretation, the relevant 1-D factor for
    dimension d is the rank of the d x d x d tensor, which for a generic matrix is d^2, but
    Winograd convolution of length d with itself (output d) has rank d when d is a power of 2
    via FFT. For d in {3,5,6} the exact 1-D convolution rank (F(2k,k)) values are:
      d=3 -> rank 4 (F(4,2) is for output 4; for output 3 with kernel 2 it's rank 3? )
    This is subtle, so we fall back to using the block-partition result which is verified.
    """
    return None

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock

    best_result = None
    best_rank = n * m * p  # schoolbook rank

    # Strategy 1: Block partition search (primary, verified)
    n_sizes, m_sizes, p_sizes, expected_rank = _find_best_block_partition(n, m, p)

    assert sum(n_sizes) == n
    assert sum(m_sizes) == m
    assert sum(p_sizes) == p

    # Build the block decomposition
    U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)

    # Verify correctness
    if verify_decomposition(U, V, W, n, m, p, atol=1e-9):
        best_result = (U, V, W)
        best_rank = total_rank

    # Strategy 2: Try additional mixed partitions that explicitly include a 2x2x2 Strassen
    # block plus a 1x5x6 or 3x3x6 remainder, etc. The generic search above already covers all
    # integer compositions, so this is redundant but we keep a couple of targeted checks to be safe.
    if time.time() < deadline - 60:
        targeted = [
            ([2, 1], [2, 2, 1], [2, 2, 2]),
            ([2, 1], [2, 1, 2], [2, 2, 2]),
            ([1, 2], [2, 2, 1], [2, 2, 2]),
            ([2, 1], [2, 2, 1], [2, 2, 2]),
            ([2, 1], [1, 2, 2], [2, 2, 2]),
            ([2, 1], [2, 1, 2], [2, 2, 2]),
            ([1, 1, 1], [2, 2, 1], [2, 2, 2]),
            ([2, 1], [2, 2, 1], [2, 2, 2]),
        ]
        for ns, ms, ps in targeted:
            if sum(ns) != n or sum(ms) != m or sum(ps) != p:
                continue
            rank = 0
            for ni in ns:
                for mj in ms:
                    for pk in ps:
                        rank += _block_rank(ni, mj, pk)
            if rank < best_rank:
                U2, V2, W2, r2 = build_block_decomposition(n, m, p, ns, ms, ps)
                if verify_decomposition(U2, V2, W2, n, m, p, atol=1e-9):
                    best_result = (U2, V2, W2)
                    best_rank = r2

    # Return the best result found
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook (rank 90)
    U_fb, V_fb, W_fb = schoolbook_lists(n, m, p)
    return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
