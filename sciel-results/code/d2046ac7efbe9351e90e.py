import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    col = 0

    # 7-term Strassen decomposition for 2x2x2 matrix multiplication
    # Each term is (U_pattern, V_pattern, W_pattern)
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def add_strassen_block(i1, j1, k1):
        nonlocal col
        for u_pat, v_pat, w_pat in strassen_terms:
            for (di, dj), coeff in u_pat.items():
                U[(i1 + di) * m + (j1 + dj)][col] += coeff
            for (dj, dk), coeff in v_pat.items():
                V[(j1 + dj) * p + (k1 + dk)][col] += coeff
            for (di, dk), coeff in w_pat.items():
                W[(k1 + dk) * n + (i1 + di)][col] += coeff
            col += 1

    def add_schoolbook_2x2x1(i1, j1, k1):
        nonlocal col
        for di in range(2):
            for dj in range(2):
                i = i1 + di
                j = j1 + dj
                k = k1
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # Partition j into two blocks of size 2, k into two blocks of size 2 plus one of size 1
    # Strassen blocks cover (i=0,1) x (j=0,1) x (k=0,1)
    add_strassen_block(0, 0, 0)
    # (i=0,1) x (j=2,3) x (k=0,1)
    add_strassen_block(0, 2, 0)
    # (i=0,1) x (j=0,1) x (k=2,3)
    add_strassen_block(0, 0, 2)
    # (i=0,1) x (j=2,3) x (k=2,3)
    add_strassen_block(0, 2, 2)
    # Schoolbook for k=4 (the 5th column)
    add_schoolbook_2x2x1(0, 0, 4)
    add_schoolbook_2x2x1(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
