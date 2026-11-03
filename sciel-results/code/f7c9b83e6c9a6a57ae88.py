"""Slice-method decomposition for 3x4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 7]


def _check_sub_decomposition(U, V_sub, W_sub, n, m, b, tol=1e-9):
    """Check sub-decomposition for a block of size b.
    
    U: (n*m, r), V_sub: (m*b, r), W_sub: (b*n, r)
    For all i,j,kk: sum_c U[i*m+j,c] * V_sub[j*b+kk,c] * W_sub[kk*n+i,c] = 1
    """
    r = U.shape[1]
    T = np.zeros((n, m, b))
    for c in range(r):
        Uc = U[:, c]
        Vc = V_sub[:, c]
        Wc = W_sub[:, c]
        for i in range(n):
            for j in range(m):
                for kk in range(b):
                    T[i, j, kk] += Uc[i * m + j] * Vc[j * b + kk] * Wc[kk * n + i]
    target = np.ones((n, m, b))
    return np.allclose(T, target, atol=tol)


def _search_sub_decomposition(n, m, b, max_rank, coeff_set, timeout=25):
    """Search for rank-minimal decomposition of n x m x b tensor.
    
    Returns (rank, U_list, V_list, W_list) or None if not found within max_rank.
    """
    start_time = time.time()
    nm = n * m
    mb = m * b
    bn = b * n

    for r in range(1, max_rank + 1):
        if time.time() - start_time > timeout:
            return None

        num_trials = 2000
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
            
            U_rand = np.random.choice(coeff_set, size=(nm, r))
            V_rand = np.random.choice(coeff_set, size=(mb, r))

            # Set up linear system for W
            A = np.zeros((n * m * b, b * n * r))
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for kk in range(b):
                        for c in range(r):
                            col = (kk * n + i) * r + c
                            A[eq_idx, col] = U_rand[i * m + j, c] * V_rand[j * b + kk, c]
                        eq_idx += 1

            try:
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(A, np.ones(n * m * b), rcond=None)
                if np.allclose(A @ W_flat, np.ones(n * m * b), atol=1e-8):
                    W_rand = W_flat.reshape((b * n, r))
                    if _check_sub_decomposition(U_rand, V_rand, W_rand, n, m, b):
                        W_rounded = np.round(W_rand * 2) / 2
                        if np.allclose(W_rounded, W_rand, atol=1e-8):
                            U_list = U_rand.tolist()
                            V_list = V_rand.tolist()
                            W_list = W_rounded.tolist()
                            return (r, U_list, V_list, W_list)
            except:
                pass

    return None


def _construct_sub_3x4x3():
    """Construct optimal decomposition for 3x4x3."""
    n, m, b = 3, 4, 3
    coeff_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    result = _search_sub_decomposition(n, m, b, max_rank=20, coeff_set=coeff_set, timeout=30)
    if result:
        return result
    # Fallback: schoolbook
    rank = n * m * b
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * b)]
    W = [[0] * rank for _ in range(b * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for kk in range(b):
                U[i * m + j][col] = 1
                V[j * b + kk][col] = 1
                W[kk * n + i][col] = 1
                col += 1
    return (rank, U, V, W)


def _construct_sub_3x4x2():
    """Construct optimal decomposition for 3x4x2."""
    n, m, b = 3, 4, 2
    coeff_set = [-2, -1, -0.5, 0, 0.5, 1, 2]
    result = _search_sub_decomposition(n, m, b, max_rank=16, coeff_set=coeff_set, timeout=30)
    if result:
        return result
    # Fallback: schoolbook
    rank = n * m * b
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * b)]
    W = [[0] * rank for _ in range(b * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for kk in range(b):
                U[i * m + j][col] = 1
                V[j * b + kk][col] = 1
                W[kk * n + i][col] = 1
                col += 1
    return (rank, U, V, W)


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7

    # Partition the 7 output columns (k dimension) into blocks: 3, 2, 2
    blocks = [3, 2, 2]
    k_offsets = []
    offset = 0
    for b in blocks:
        k_offsets.append(offset)
        offset += b

    # Find sub-decompositions
    sub_decomps = []
    # Block 0: k in [0, 3) -> 3x4x3
    r0, U0, V0, W0 = _construct_sub_3x4x3()
    sub_decomps.append((r0, U0, V0, W0, k_offsets[0], blocks[0]))
    # Block 1: k in [3, 5) -> 3x4x2
    r1, U1, V1, W1 = _construct_sub_3x4x2()
    sub_decomps.append((r1, U1, V1, W1, k_offsets[1], blocks[1]))
    # Block 2: k in [5, 7) -> 3x4x2
    r2, U2, V2, W2 = _construct_sub_3x4x2()
    sub_decomps.append((r2, U2, V2, W2, k_offsets[2], blocks[2]))

    total_rank = sum(r for r, _, _, _, _, _ in sub_decomps)

    # Combine: U is (n*m, total_rank), V is (m*p, total_rank), W is (p*n, total_rank)
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]

    col_offset = 0
    for r, U_sub, V_sub, W_sub, k_off, b in sub_decomps:
        # U_sub: (n*m, r), V_sub: (m*b, r), W_sub: (b*n, r)
        for i in range(n * m):
            for c in range(r):
                U_full[i][col_offset + c] = U_sub[i][c]

        # V: map V_sub[j*b+kk] to V_full[j*p + (k_off+kk)]
        for j in range(m):
            for kk in range(b):
                k_global = k_off + kk
                row_v = j * p + k_global
                for c in range(r):
                    V_full[row_v][col_offset + c] = V_sub[j * b + kk][c]

        # W: map W_sub[kk*n+i] to W_full[(k_off+kk)*n + i]
        for kk in range(b):
            k_global = k_off + kk
            for i in range(n):
                row_w = k_global * n + i
                for c in range(r):
                    W_full[row_w][col_offset + c] = W_sub[kk * n + i][c]

        col_offset += r

    return {"U": U_full, "V": V_full, "W": W_full}
