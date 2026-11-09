"""Matrix multiplication 4,4,9: tensor decomposition with rank minimization."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    # Schoolbook decomposition: rank = n*m*p = 144
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_rank = rank_school
    best_result = {"U": U, "V": V, "W": W}

    def verify_full(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Full verification: check all (a,b,c) triples."""
        num_cols = len(U_mat[0]) if U_mat else 0
        if num_cols == 0:
            return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (j == j2 and k == k2 and i == i2) else 0.0
                    s = 0.0
                    for r in range(num_cols):
                        s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
                    if abs(s - expected) > tol:
                        return False
        return True

    def verify_fast(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Faster verification: check all non-zero entries and a sample of zeros."""
        num_cols = len(U_mat[0]) if U_mat else 0
        if num_cols == 0:
            return False
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(num_cols):
                        s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
                    if abs(s - 1.0) > tol:
                        return False
        # Check a sample of zero entries
        import random
        random.seed(42)
        zero_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    zero_set.add((i * m + j, j * p + k, k * n + i))
        checked = 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in zero_set:
                        continue
                    s = 0.0
                    for r in range(num_cols):
                        s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
                    if abs(s) > tol:
                        return False
                    checked += 1
                    if checked >= 8000:
                        break
                if checked >= 8000:
                    break
            if checked >= 8000:
                break
        return True

    # Strassen 2x2x2 rank-7 decomposition coefficients
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, -1, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    def build_4x4x4_strassen():
        """Build rank-56 decomposition for 4x4*4x4 using 8 Strassen 2x2x2 blocks.
        Returns (U_cols, V_cols, W_cols) each 16x56 lists.
        """
        U_cols = [[0] * 56 for _ in range(16)]
        V_cols = [[0] * 56 for _ in range(16)]
        W_cols = [[0] * 56 for _ in range(16)]

        products = [
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (0, 0, 0, 1),
            (0, 1, 1, 1),
            (1, 0, 0, 0),
            (1, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
        ]

        for prod_idx, (ar, ac, br, bc) in enumerate(products):
            for s in range(7):
                col = prod_idx * 7 + s
                for i in range(2):
                    for j in range(2):
                        U_cols[(ar * 2 + i) * 4 + (ac * 2 + j)][col] = strassen_U[s][i * 2 + j]
                        V_cols[(br * 2 + i) * 4 + (bc * 2 + j)][col] = strassen_V[s][i * 2 + j]
                        W_cols[(bc * 2 + j) * 4 + (ar * 2 + i)][col] = strassen_W[s][i * 2 + j]

        return U_cols, V_cols, W_cols

    # Build rank-128 decomposition: 56 + 56 + 16
    if time.time() < deadline - 30:
        total_rank = 128
        U_new = [[0] * total_rank for _ in range(16)]
        V_new = [[0] * total_rank for _ in range(36)]
        W_new = [[0] * total_rank for _ in range(36)]

        # Block 1: cols 0-3 of B (4x4 sub-problem), rank 56
        U1, V1, W1 = build_4x4x4_strassen()
        for r in range(56):
            for a in range(16):
                U_new[a][r] = U1[a][r]
            for b in range(16):
                j = b // 4
                k = b % 4
                V_new[j * 9 + k][r] = V1[b][r]
            for c in range(16):
                W_new[c][r] = W1[c][r]

        # Block 2: cols 4-7 of B (4x4 sub-problem), rank 56
        U2, V2, W2 = build_4x4x4_strassen()
        for r in range(56):
            for a in range(16):
                U_new[a][56 + r] = U2[a][r]
            for b in range(16):
                j = b // 4
                k = b % 4
                V_new[j * 9 + (4 + k)][56 + r] = V2[b][r]
            for c in range(16):
                k_c = c // 4
                i_c = c % 4
                W_new[(4 + k_c) * 4 + i_c][56 + r] = W2[c][r]

        # Block 3: col 8 of B (4x1 sub-problem), rank 16 (schoolbook)
        for i in range(4):
            for j in range(4):
                r = 112 + i * 4 + j
                U_new[i * 4 + j][r] = 1
                V_new[j * 9 + 8][r] = 1
                W_new[8 * 4 + i][r] = 1

        if verify_fast(U_new, V_new, W_new, n, m, p):
            best_rank = 128
            best_result = {"U": U_new, "V": V_new, "W": W_new}

    # Try an alternative split: 9 = 5 + 4
    # 4x4x5: use Strassen for 4x4x4 part (rank 56) + schoolbook for 4x4x1 (rank 16) = 72
    # 4x4x4: Strassen rank 56
    # Total: 72 + 56 = 128 (same as before, but let's verify)
    if time.time() < deadline - 20:
        total_rank2 = 128
        U_new2 = [[0] * total_rank2 for _ in range(16)]
        V_new2 = [[0] * total_rank2 for _ in range(36)]
        W_new2 = [[0] * total_rank2 for _ in range(36)]

        # Block A: cols 0-4 of B (4x5 sub-problem)
        # 4x4x4 part (cols 0-3), rank 56
        U1, V1, W1 = build_4x4x4_strassen()
        for r in range(56):
            for a in range(16):
                U_new2[a][r] = U1[a][r]
            for b in range(16):
                j = b // 4
                k = b % 4
                V_new2[j * 9 + k][r] = V1[b][r]
            for c in range(16):
                W_new2[c][r] = W1[c][r]
        # 4x4x1 part (col 4), rank 16
        for i in range(4):
            for j in range(4):
                r = 56 + i * 4 + j
                U_new2[i * 4 + j][r] = 1
                V_new2[j * 9 + 4][r] = 1
                W_new2[4 * 4 + i][r] = 1

        # Block B: cols 5-8 of B (4x4 sub-problem), rank 56
        U2, V2, W2 = build_4x4x4_strassen()
        for r in range(56):
            for a in range(16):
                U_new2[a][72 + r] = U2[a][r]
            for b in range(16):
                j = b // 4
                k = b % 4
                V_new2[j * 9 + (5 + k)][72 + r] = V2[b][r]
            for c in range(16):
                k_c = c // 4
                i_c = c % 4
                W_new2[(5 + k_c) * 4 + i_c][72 + r] = W2[c][r]

        if verify_fast(U_new2, V_new2, W_new2, n, m, p):
            best_rank = 128
            best_result = {"U": U_new2, "V": V_new2, "W": W_new2}

    # Try split: 9 = 6 + 3
    # 4x4x6: Strassen 4x4x4 (56) + schoolbook 4x4x2 (32) = 88
    # 4x4x3: schoolbook 48
    # Total: 88 + 48 = 136 (worse, skip)

    # Try split: 9 = 4 + 3 + 2
    # 4x4x4: Strassen 56
    # 4x4x3: schoolbook 48
    # 4x4x2: schoolbook 32
    # Total: 56 + 48 + 32 = 136 (worse, skip)

    # Try a different approach: use Strassen for overlapping blocks
    # Actually, let's try to use a rank-7 decomposition more cleverly.
    # The key is that Strassen reduces 2x2x2 from 8 to 7.
    # For 4x4x9, we have 4*4*9 = 144 schoolbook.
    # With Strassen: 8 blocks of 2x2x2 for the 4x4x4 part (56 each, 2 blocks = 112)
    # Plus 4x4x1 part (16) = 128.

    # Let's try to see if we can do better with a different block structure.
    # What if we use Strassen for 2x2x2 blocks in a more efficient way?
    # For 4x4x9, we can think of it as 2x(2x4x9) or 4x(1x4x9), etc.

    # Actually, the current best is 128. Let's try to verify it's correct and return it.

    return best_result

# EVOLVE-BLOCK-END
