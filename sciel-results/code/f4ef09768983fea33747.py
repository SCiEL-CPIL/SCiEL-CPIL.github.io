import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_DEADLINE = time.time() + 145.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    # Block size 3, so 2x2 blocks
    block_size = 3
    blocks_per_dim = n // block_size  # 2

    # Strassen patterns for 2x2 blocks
    # Each pattern: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict (block_row, block_col) -> value
    # B_coeffs: dict (block_row, block_col) -> value
    # C_coeffs: dict (block_row, block_col) -> value
    #
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    #
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5

    strassen_patterns = [
        # M0
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1},
         {(0, 0): 1, (1, 1): 1}),
        # M1
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1},
         {(1, 0): 1, (1, 1): -1}),
        # M2
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1},
         {(0, 1): 1, (1, 1): 1}),
        # M3
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1},
         {(0, 0): 1, (1, 0): 1}),
        # M4
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1},
         {(0, 0): -1, (0, 1): 1}),
        # M5
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1},
         {(1, 1): 1}),
        # M6
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1},
         {(0, 0): 1}),
    ]

    rank = 7 * (block_size ** 3)  # 7 * 27 = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col_idx = 0
    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
        # For each local index i, j, k in the block
        for i in range(block_size):
            for j in range(block_size):
                for k in range(block_size):
                    # U contributions
                    for (br, bc), val in u_coeffs.items():
                        row = br * block_size + i
                        col = bc * block_size + j
                        U[row * m + col][col_idx] += val

                    # V contributions
                    for (br, bc), val in v_coeffs.items():
                        row = br * block_size + j
                        col = bc * block_size + k
                        V[row * p + col][col_idx] += val

                    # W contributions
                    for (br, bc), val in w_coeffs.items():
                        # W index: k_global * n + i_global
                        # k_global = bc * block_size + k
                        # i_global = br * block_size + i
                        k_global = bc * block_size + k
                        i_global = br * block_size + i
                        W[k_global * n + i_global][col_idx] += val

                    col_idx += 1

    # Verification
    if time.time() < _DEADLINE - 15.0:
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        
        Ur = Ua.reshape(n, m, rank)
        Vr = Va.reshape(m, p, rank)
        Wr = Wa.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        expected = np.ones((n, m, p), dtype=np.int64)
        
        if not np.array_equal(T, expected):
            # Fallback to naive
            rank_t = n * m * p
            Ut = [[0] * rank_t for _ in range(n * m)]
            Vt = [[0] * rank_t for _ in range(m * p)]
            Wt = [[0] * rank_t for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = i * m * p + j * p + k
                        Ut[i * m + j][r] = 1
                        Vt[j * p + k][r] = 1
                        Wt[k * n + i][r] = 1
            return {"U": Ut, "V": Vt, "W": Wt}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
