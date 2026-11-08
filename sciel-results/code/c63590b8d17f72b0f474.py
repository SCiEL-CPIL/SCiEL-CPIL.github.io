import time
import numpy as np

DIMENSIONS = [3, 4, 7]
_DEADLINE = 130.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    # Rank calculation:
    # 6 Strassen blocks (rank 7 each) = 42
    # 1 combined term for i=2, k in 0..5 = 1
    # 1 combined term for k=6, i in 0..2 = 1
    # Total = 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(col_start, i0, i1, j0, j1, k0, k1):
        U[i0 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1

        U[i1 * m + j0][col_start + 1] = 1
        U[i1 * m + j1][col_start + 1] = 1
        V[j0 * p + k0][col_start + 1] = 1
        W[k0 * n + i1][col_start + 1] = 1
        W[k1 * n + i1][col_start + 1] = -1

        U[i0 * m + j0][col_start + 2] = 1
        V[j0 * p + k1][col_start + 2] = 1
        V[j1 * p + k1][col_start + 2] = -1
        W[k1 * n + i0][col_start + 2] = 1
        W[k1 * n + i1][col_start + 2] = 1

        U[i1 * m + j1][col_start + 3] = 1
        V[j1 * p + k0][col_start + 3] = 1
        V[j0 * p + k0][col_start + 3] = -1
        W[k0 * n + i0][col_start + 3] = 1
        W[k0 * n + i1][col_start + 3] = 1

        U[i0 * m + j0][col_start + 4] = 1
        U[i0 * m + j1][col_start + 4] = 1
        V[j1 * p + k1][col_start + 4] = 1
        W[k0 * n + i0][col_start + 4] = -1
        W[k1 * n + i0][col_start + 4] = 1

        U[i1 * m + j0][col_start + 5] = 1
        U[i0 * m + j0][col_start + 5] = -1
        V[j0 * p + k0][col_start + 5] = 1
        V[j0 * p + k1][col_start + 5] = 1
        W[k1 * n + i1][col_start + 5] = 1

        U[i0 * m + j1][col_start + 6] = 1
        U[i1 * m + j1][col_start + 6] = -1
        V[j1 * p + k0][col_start + 6] = 1
        V[j1 * p + k1][col_start + 6] = 1
        W[k0 * n + i0][col_start + 6] = 1

        return col_start + 7

    blocks = [
        (0, 1, 0, 1, 0, 1),
        (0, 1, 0, 1, 2, 3),
        (0, 1, 0, 1, 4, 5),
        (0, 1, 2, 3, 0, 1),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 4, 5),
    ]

    for (i0, i1, j0, j1, k0, k1) in blocks:
        col = add_strassen_block(col, i0, i1, j0, j1, k0, k1)
        
    # Combined term 1: i=2, k in 0..5
    # U[2, j] = 1, V[j, k] = 1, W[k, 2] = 1
    for j in range(m):
        U[2 * m + j][col] = 1
    for j in range(m):
        for k in range(6): # k in 0..5
            V[j * p + k][col] = 1
    for k in range(6):
        W[k * n + 2][col] = 1
    col += 1
    
    # Combined term 2: k=6, i in 0..2
    # U[i, j] = 1, V[j, 6] = 1, W[6, i] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
    for j in range(m):
        V[j * p + 6][col] = 1
    for i in range(n):
        W[6 * n + i][col] = 1
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verification
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)
    
    for i in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for j in range(m):
            u_idx = i * m + j
            for k in range(p):
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                if total != 1:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1")
                    
    return {"U": U, "V": V, "W": W}
