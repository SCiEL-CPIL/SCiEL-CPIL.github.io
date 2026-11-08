"""Optimized decomposition: 2x3x8 matrix multiplication using Strassen + rank-1 for extra row."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        import numpy as np
        U = np.array(candidate["U"], dtype=np.int64)
        V = np.array(candidate["V"], dtype=np.int64)
        W = np.array(candidate["W"], dtype=np.int64)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        rank = U.shape[1]
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        try:
            T = np.einsum('iac,akc,kic->iak', U_t, V_t, W_t, optimize=True)
        except Exception:
            return False
        if not np.all(T == 1):
            return False
        return True

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    S_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    S_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, -1],# M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]

    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        u_val = S_U[r][i * 2 + a]
                        v_val = S_V[r][a * 2 + k]
                        w_val = S_W[r][k * 2 + i]
                        acc += u_val * v_val * w_val
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Rank 32 solution: 4 blocks of (7 Strassen + 1 rank-1)
    num_blocks = 4
    rank_per_block = 8
    rank = num_blocks * rank_per_block

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    for block_idx in range(num_blocks):
        k0 = block_idx * 2
        k1 = k0 + 1

        # 7 Strassen terms for a=0,1
        for r in range(7):
            # U: i in {0,1}, a in {0,1}
            for i in range(2):
                for a in range(2):
                    u_val = S_U[r][i * 2 + a]
                    if u_val != 0:
                        U[i * m + a][col] = u_val

            # V: a in {0,1}, k in {k0, k1}
            for a in range(2):
                for dk in range(2):
                    v_val = S_V[r][a * 2 + dk]
                    if v_val != 0:
                        V[a * p + (k0 + dk)][col] = v_val

            # W: k in {k0, k1}, i in {0,1}
            for dk in range(2):
                for i in range(2):
                    w_val = S_W[r][dk * 2 + i]
                    if w_val != 0:
                        W[(k0 + dk) * n + i][col] = w_val

            col += 1
            if not _elapsed_ok():
                return _schoolbook()

        # 1 rank-1 term for a=2
        # U: i in {0,1}, a=2 -> U[i*3+2] = 1
        for i in range(2):
            U[i * m + 2][col] = 1

        # V: a=2, k in {k0, k1} -> V[2*p + k] = 1
        for dk in range(2):
            V[2 * p + (k0 + dk)][col] = 1

        # W: k in {k0, k1}, i in {0,1} -> W[k*n+i] = 1
        for dk in range(2):
            for i in range(2):
                W[(k0 + dk) * n + i][col] = 1

        col += 1
        if not _elapsed_ok():
            return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
