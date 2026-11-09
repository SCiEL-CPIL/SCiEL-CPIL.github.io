"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 100.0

    # Precomputed 2x2x2 Strassen factors (rank 7).
    # U_s rows: A00, A01, A10, A11  (index = i*2 + j)
    # V_s rows: B00, B01, B10, B11  (index = j*2 + k)
    # W_s rows: C00, C01, C10, C11  (index = k*2 + i)
    U_s = np.array([
        [1, 1, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 1, -1, 1, 0, 0, -1]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Build rank-42 decomposition using 6 independent Strassen blocks.
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for a_half in range(2):
        for b_col_block in range(3):
            b_row_half = a_half
            col_off = (a_half * 3 + b_col_block) * 7

            for r in range(7):
                for idx in range(4):
                    # U: A[i, a_col] where idx = i*2 + local_j
                    i = idx // 2
                    lj = idx % 2
                    a_col = a_half * 2 + lj
                    U[i * m + a_col, col_off + r] = U_s[idx, r]

                    # V: B[b_row, b_col] where idx = local_j*2 + local_k
                    b_row = b_row_half * 2 + idx // 2
                    b_col = b_col_block * 2 + idx % 2
                    V[b_row * p + b_col, col_off + r] = V_s[idx, r]

                    # W: C[c_col, c_row] where idx = local_k*2 + local_i
                    c_row = idx % 2
                    c_col = b_col_block * 2 + idx // 2
                    W[c_col * n + c_row, col_off + r] = W_s[idx, r]

    # Quick verification with a single random test.
    rng = np.random.default_rng(42)
    A = rng.integers(-5, 6, size=(n, m))
    B = rng.integers(-5, 6, size=(m, p))
    C_direct = A @ B
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(rank):
                    s += U[i*m+j, r] * A[i, j] * V[j*p+k, r] * B[j, k] * W[k*n+i, r]
            C_decomp[i, k] = s
    if not np.allclose(C_decomp, C_direct, atol=1e-9):
        # Fallback: schoolbook rank 48
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
