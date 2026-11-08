import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 120  # safety margin

    n, m, p = DIMENSIONS  # 4, 4, 6

    # Strassen 2x2x2 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]

    # 12-block 2x2x2 tiling: split 4x4x6 into 12 sub-problems of 2x2x2
    # Each sub-problem uses Strassen rank-7, total rank = 12 * 7 = 84
    total_rank = 84
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * total_rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 84

    # Block indices: bi in {0,1} (row block of A), bj in {0,1} (col block of A / row block of B), bk in {0,1,2} (col block of B)
    for bi in range(2):
        for bj in range(2):
            for bk in range(3):
                # Base column index for this 2x2x2 block
                base_col = (bi * 2 + bj) * 3 + bk

                # Fill U: rows (i, j) where i = bi*2 + di, j = bj*2 + dj
                for di in range(2):
                    for dj in range(2):
                        i = bi * 2 + di
                        j = bj * 2 + dj
                        row_U = i * m + j
                        for c in range(7):
                            col = base_col * 7 + c
                            U[row_U][col] = S_U[di * 2 + dj][c]

                # Fill V: rows (j, k) where j = bj*2 + dj, k = bk*2 + dk
                for dj in range(2):
                    for dk in range(2):
                        j = bj * 2 + dj
                        k = bk * 2 + dk
                        row_V = j * p + k
                        for c in range(7):
                            col = base_col * 7 + c
                            V[row_V][col] = S_V[dj * 2 + dk][c]

                # Fill W: rows (k, i) where k = bk*2 + dk, i = bi*2 + di
                for dk in range(2):
                    for di in range(2):
                        k = bk * 2 + dk
                        i = bi * 2 + di
                        row_W = k * n + i
                        for c in range(7):
                            col = base_col * 7 + c
                            W[row_W][col] = S_W[dk * 2 + di][c]

    # Remove zero columns to reduce rank
    keep_cols = []
    for c in range(total_rank):
        if any(U[r][c] != 0 for r in range(n * m)) or \
           any(V[r][c] != 0 for r in range(m * p)) or \
           any(W[r][c] != 0 for r in range(p * n)):
            keep_cols.append(c)

    final_rank = len(keep_cols)
    U_final = [[U[r][c] for c in keep_cols] for r in range(n * m)]
    V_final = [[V[r][c] for c in keep_cols] for r in range(m * p)]
    W_final = [[W[r][c] for c in keep_cols] for r in range(p * n)]

    # Time guard: return immediately since construction is O(1)
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Fallback: return schoolbook (rank 96)
        U_fb = [[1 if (i*m+j) == r else 0 for r in range(n*m*p)] for i in range(n) for j in range(m)]
        V_fb = [[1 if (j*p+k) == r else 0 for r in range(n*m*p)] for j in range(m) for k in range(p)]
        W_fb = [[1 if (k*n+i) == r else 0 for r in range(n*m*p)] for k in range(p) for i in range(n)]
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U_final, "V": V_final, "W": W_final}
