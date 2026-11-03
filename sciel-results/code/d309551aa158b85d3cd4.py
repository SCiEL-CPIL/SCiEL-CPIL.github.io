import numpy as np
import time

DIMENSIONS = [4, 4, 5]


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Verify using numpy einsum for speed."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, rank)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, rank)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p), dtype=np.float64)
    return np.allclose(expected, reconstructed, atol=tol)


def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _strassen_2x2():
    """Rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where indices are (row, col) for 2x2.
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # Term 0: m0 = (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # Term 1: m1 = (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # Term 2: m2 = a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # Term 3: m3 = a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1

    # Term 4: m4 = (a00+a01)b11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = -1; W[3][4] = 1

    # Term 5: m5 = (a10-a00)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # Term 6: m6 = (a01-a11)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4 x 4x4 using 2x2 Strassen blocks.
    Split 4x4 into four 2x2 blocks. Use Strassen on the block level,
    and schoolbook (rank 4) for the inner 2x2 multiplications.
    Total rank = 7 * 4 = 28.
    """
    # For each of the 7 Strassen terms, define which (block_row, block_col)
    # pairs of A, B, C are used and their coefficients.
    block_coeffs_U = [
        [(0, 0, 1), (1, 1, 1)],      # m0
        [(1, 0, 1), (1, 1, 1)],      # m1
        [(0, 0, 1)],                 # m2
        [(1, 1, 1)],                 # m3
        [(0, 0, 1), (0, 1, 1)],      # m4
        [(1, 0, 1), (0, 0, -1)],     # m5
        [(0, 1, 1), (1, 1, -1)],     # m6
    ]
    block_coeffs_V = [
        [(0, 0, 1), (1, 1, 1)],      # m0
        [(0, 0, 1)],                 # m1
        [(0, 1, 1), (1, 1, -1)],     # m2
        [(1, 0, 1), (0, 0, -1)],     # m3
        [(1, 1, 1)],                 # m4
        [(0, 0, 1), (0, 1, 1)],      # m5
        [(1, 0, 1), (1, 1, 1)],      # m6
    ]
    block_coeffs_W = [
        [(0, 0, 1), (1, 1, 1)],      # m0
        [(1, 0, 1), (1, 1, -1)],     # m1
        [(0, 1, 1), (1, 1, 1)],      # m2
        [(1, 0, 1), (0, 0, 1)],      # m3
        [(0, 1, -1), (1, 1, 1)],     # m4
        [(1, 1, 1)],                 # m5
        [(0, 0, 1)],                 # m6
    ]

    rank = 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]

    for s in range(7):
        for lr in range(2):   # local row within 2x2 block
            for lc in range(2):  # local col within 2x2 block
                c = s * 4 + lr * 2 + lc

                # Inner schoolbook: local row/col weight is 1 for each (lr2, lc2)
                for (br, bc, coeff_u) in block_coeffs_U[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            i = br * 2 + lr2
                            j = bc * 2 + lc2
                            U[i * 4 + j][c] += coeff_u * 1

                for (br, bc, coeff_v) in block_coeffs_V[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            j = br * 2 + lr2
                            k = bc * 2 + lc2
                            V[j * 4 + k][c] += coeff_v * 1

                for (br, bc, coeff_w) in block_coeffs_W[s]:
                    for lr2 in range(2):
                        for lc2 in range(2):
                            k = br * 2 + lr2
                            i = bc * 2 + lc2
                            W[k * 4 + i][c] += coeff_w * 1

    return U, V, W


def _try_reduce_rank(U, V, W, n, m, p, max_iter=200, tol=1e-9):
    """Attempt to reduce rank by removing provably-redundant rank-1 terms.
    Returns (U, V, W, new_rank) if reduction succeeds, else None.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    n, m, p = n, m, p

    for _ in range(max_iter):
        removed = False
        for c in range(rank - 1, -1, -1):
            # Check if removing column c still satisfies the decomposition
            U_test = [row[:c] + row[c + 1:] for row in U]
            V_test = [row[:c] + row[c + 1:] for row in V]
            W_test = [row[:c] + row[c + 1:] for row in W]
            if _verify_decomposition_fast(U_test, V_test, W_test, n, m, p, tol):
                U = U_test
                V = V_test
                W = W_test
                rank -= 1
                removed = True
                break
        if not removed:
            break

    if rank < len(U[0]) or rank < len(V[0]) or rank < len(W[0]):
        return U, V, W, rank
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    if not _verify_decomposition_fast(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook failed")

    best_rank = n * m * p
    best_result = {"U": U_school, "V": V_school, "W": W_school}

    # Build structured 4x4x4 Strassen + boundary terms for the 5th column
    if time.time() < deadline:
        U44, V44, W44 = _build_4x4x4_strassen()
        # Verify the 4x4x4 part
        if _verify_decomposition_fast(U44, V44, W44, 4, 4, 4):
            rank_core = 28
            rank_boundary = 16  # 4x4 matrix-vector multiply for 5th column
            rank_total = rank_core + rank_boundary

            U_new = [[0] * rank_total for _ in range(16)]
            V_new = [[0] * rank_total for _ in range(20)]  # 4*5
            W_new = [[0] * rank_total for _ in range(20)]  # 5*4

            # 4x4x4 Strassen part: A(4x4) * B_left(4x4) -> C_left(4x4)
            # U[i,j,c] for i,j in 0..3, c in 0..27
            # V[j,k,c] for j in 0..3, k in 0..3, c in 0..27 (k=4 is 0)
            # W[k,i,c] for k in 0..3, i in 0..3, c in 0..27 (k=4 is 0)
            for c in range(rank_core):
                for i in range(4):
                    for j in range(4):
                        U_new[i * 4 + j][c] = U44[i * 4 + j][c]
                for j in range(4):
                    for k in range(4):
                        V_new[j * 5 + k][c] = V44[j * 4 + k][c]
                for k in range(4):
                    for i in range(4):
                        W_new[k * 4 + i][c] = W44[k * 4 + i][c]

            # Boundary: A(4x4) * B_right(4x1) -> C_right(4x1)
            # C_right[i,4] = sum_j A[i,j]*B[j,4]
            # For each (i,j): term c = 28 + i*4 + j
            for i in range(4):
                for j in range(4):
                    c = rank_core + i * 4 + j
                    U_new[i * 4 + j][c] = 1
                    V_new[j * 5 + 4][c] = 1
                    W_new[4 * 4 + i][c] = 1

            # Verify
            if _verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
                best_rank = rank_total
                best_result = {"U": U_new, "V": V_new, "W": W_new}

                # Attempt rank reduction via removing redundant terms
                if time.time() < deadline:
                    reduced = _try_reduce_rank(U_new, V_new, W_new, n, m, p, max_iter=300)
                    if reduced is not None:
                        U_r, V_r, W_r, rank_r = reduced
                        if rank_r < best_rank and _verify_decomposition_fast(U_r, V_r, W_r, n, m, p):
                            best_rank = rank_r
                            best_result = {"U": U_r, "V": V_r, "W": W_r}

    # If we still have time and haven't found a great decomposition, try a few
    # random structured perturbations around the Strassen-based solution.
    if time.time() < deadline and best_rank > 40:
        import random
        rng = random.Random(12345)
        target_ranks = [42, 40, 38, 36, 32, 28, 24, 20, 16, 12, 8]

        for target_rank in target_ranks:
            if time.time() > deadline:
                break
            if target_rank >= best_rank:
                continue

            attempts = min(30, int((deadline - time.time()) / 0.5))
            for _ in range(attempts):
                if time.time() > deadline:
                    break

                U = [[0] * target_rank for _ in range(16)]
                V = [[0] * target_rank for _ in range(20)]
                W = [[0] * target_rank for _ in range(20)]

                choices = [-1, -0.5, 0, 0.5, 1]
                for c in range(target_rank):
                    for i in range(16):
                        U[i][c] = rng.choice(choices)
                    for i in range(20):
                        V[i][c] = rng.choice(choices)
                        W[i][c] = rng.choice(choices)

                if _verify_decomposition_fast(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break

    if time.time() > deadline:
        return best_result

    return best_result
