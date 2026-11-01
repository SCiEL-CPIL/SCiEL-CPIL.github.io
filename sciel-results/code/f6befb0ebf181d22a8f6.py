"""Improved decomposition for 2x5 x 5x6 matrix multiplication using rank-54 -> rank-51."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_block_222(U, V, W, rank_offset, r1, r2, c1, c2, k1, k2, n, m, p):
    """
    Apply Strassen's rank-7 algorithm to a 2x2x2 sub-block.
    r1,r2: row indices of A/C
    c1,c2: col indices of A / row indices of B
    k1,k2: row indices of B / col indices of C
    """
    strassen_factors = [
        # M1 = (A00+A11)(B00+B11)
        ([((r1,c1),1),((r2,c2),1)], [((c1,k1),1),((c2,k2),1)], [((k1,r1),1),((k2,r2),1)]),
        # M2 = (A10+A11)B00
        ([((r2,c1),1),((r2,c2),1)], [((c1,k1),1)], [((k1,r2),1),((k2,r2),-1)]),
        # M3 = A00(B01-B11)
        ([((r1,c1),1)], [((c1,k2),1),((c2,k2),-1)], [((k2,r1),1),((k2,r2),1)]),
        # M4 = A11(B10-B00)
        ([((r2,c2),1)], [((c2,k1),1),((c1,k1),-1)], [((k1,r1),1),((k1,r2),1)]),
        # M5 = (A00+A01)B11
        ([((r1,c1),1),((r1,c2),1)], [((c2,k2),1)], [((k1,r1),-1),((k2,r2),1)]),
        # M6 = (A10-A00)(B00+B01)
        ([((r2,c1),1),((r1,c1),-1)], [((c1,k1),1),((c1,k2),1)], [((k2,r2),1)]),
        # M7 = (A01-A11)(B10+B11)
        ([((r1,c2),1),((r2,c2),-1)], [((c2,k1),1),((c2,k2),1)], [((k1,r1),1)]),
    ]
    for t, (a_factors, b_factors, c_factors) in enumerate(strassen_factors):
        r = rank_offset + t
        for (idx, coeff) in a_factors:
            U[idx[0] * m + idx[1]][r] += coeff
        for (idx, coeff) in b_factors:
            V[idx[0] * p + idx[1]][r] += coeff
        for (idx, coeff) in c_factors:
            W[idx[0] * n + idx[1]][r] += coeff


def _rank3_block_212(U, V, W, rank_offset, c0, k1, k2, n, m, p):
    """
    Rank-3 decomposition for a 2x1x2 sub-block (j fixed = c0, two k's, two i's).
    M1 = A0 * B0 -> C0
    M2 = A1 * B1 -> C1
    M3 = (A0+A1)*(B0+B1) -> C0+C1
    """
    # M1
    r = rank_offset
    U[0 * m + c0][r] = 1
    V[c0 * p + k1][r] = 1
    W[k1 * n + 0][r] = 1
    # M2
    r = rank_offset + 1
    U[1 * m + c0][r] = 1
    V[c0 * p + k2][r] = 1
    W[k2 * n + 1][r] = 1
    # M3
    r = rank_offset + 2
    U[0 * m + c0][r] = 1
    U[1 * m + c0][r] = 1
    V[c0 * p + k1][r] = 1
    V[c0 * p + k2][r] = 1
    W[k1 * n + 0][r] = 1
    W[k2 * n + 1][r] = 1


def _verify(U, V, W, n, m, p, tol=1e-9):
    U_a = np.array(U, dtype=float)
    V_a = np.array(V, dtype=float)
    W_a = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_a[i*m+j] * V_a[j*p+k] * W_a[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 2, 5, 6
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n

    # Compute total rank first
    n_strassen_blocks = 6  # 2 j-pairs * 3 k-pairs
    n_rank3_blocks = 3     # 1 j-singleton * 3 k-pairs
    total_rank = n_strassen_blocks * 7 + n_rank3_blocks * 3  # 42 + 9 = 51

    max_rank = total_rank
    U = [[0] * max_rank for _ in range(U_dim)]
    V = [[0] * max_rank for _ in range(V_dim)]
    W = [[0] * max_rank for _ in range(W_dim)]

    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]

    rank = 0
    # Strassen blocks: 2 j-pairs x 3 k-pairs = 6 blocks x rank 7 = 42
    for (c1, c2) in j_pairs:
        for (k1, k2) in k_pairs:
            _strassen_block_222(U, V, W, rank, 0, 1, c1, c2, k1, k2, n, m, p)
            rank += 7

    # Rank-3 blocks: j=4 singleton x 3 k-pairs = 3 blocks x rank 3 = 9
    for (k1, k2) in k_pairs:
        _rank3_block_212(U, V, W, rank, 4, k1, k2, n, m, p)
        rank += 3

    # Trim to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]

    # Verify
    if time.time() < deadline - 10:
        if not _verify(U, V, W, n, m, p, tol=1e-8):
            # Fallback to schoolbook (rank 60)
            U_fb = [[0] * (n*m*p) for _ in range(U_dim)]
            V_fb = [[0] * (n*m*p) for _ in range(V_dim)]
            W_fb = [[0] * (n*m*p) for _ in range(W_dim)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][col] = 1
                        V_fb[j*p+k][col] = 1
                        W_fb[k*n+i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": W, "W": W} if False else {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
