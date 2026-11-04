"""Rank-28 tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses Winograd F(4) for the 4x4x4 block and schoolbook for the 4x4x1 block.
"""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5

    # Winograd F(4) transform matrices
    M = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ]
    D = [
        [1, 0, 0, 0],
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1],
    ]
    Wt = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
    ]

    U_cols = []  # Each element is a column of U (length n*m = 16)
    V_cols = []  # Each element is a column of V (length m*p = 20)
    W_cols = []  # Each element is a column of W (length p*n = 20)

    # Part 1: Winograd F(4) for 4x4x4 block (columns 0..3 of B)
    for i in range(4):
        for k in range(4):
            if D[i][k] == 0:
                continue
            # U[a,j] = M[a,i] for a in 0..3, j in 0..3
            U_vec = [0] * (n * m)
            V_vec = [0] * (m * p)
            W_vec = [0] * (p * n)
            for a in range(4):
                for j in range(4):
                    U_vec[a * m + j] = M[a][i]
            for j in range(4):
                for b in range(4):
                    V_vec[j * p + b] = Wt[b][k]
            for q in range(4):
                for p_idx in range(4):
                    W_vec[q * n + p_idx] = D[i][k] * M[p_idx][i] * Wt[k][q]
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)

    # Part 2: Schoolbook for 4x4x1 block (column 4 of B)
    for i in range(4):
        for j in range(4):
            U_vec = [0] * (n * m)
            V_vec = [0] * (m * p)
            W_vec = [0] * (p * n)
            U_vec[i * m + j] = 1
            V_vec[j * p + 4] = 1
            W_vec[4 * n + i] = 1
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)

    rank = len(U_cols)
    # Transpose: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
