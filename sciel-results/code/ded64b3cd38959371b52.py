import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # A is n x m = 2x4, B is m x p = 4x6, C is p x n = 6x2

    rank = 42

    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    # Strassen coefficient matrices: rows indexed by (local_row, local_col),
    # columns indexed by term s in 0..6
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]

    strassen_V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]

    # C00 = M1 - M4 + M5 - M7
    # C01 = -M3 - M5
    # C10 = -M2 + M4
    # C11 = M1 + M2 - M3 - M6
    strassen_W = [
        [1, 0, 0, -1, 1, 0, -1],  # C00
        [0, 0, -1, 0, -1, 0, 0],  # C01
        [0, -1, 0, 1, 0, 0, 0],   # C10
        [1, 1, -1, 0, 0, -1, 0]   # C11
    ]

    col = 0

    # A is 2x4, split into 2 blocks of 2x2 along columns:
    #   A_block a=0: columns 0,1 (2x2)
    #   A_block a=1: columns 2,3 (2x2)
    # B is 4x6, split into 4 blocks of 2x2:
    #   B block (br, bc): rows br*2..br*2+1, cols bc*2..bc*2+1
    #   A_block a multiplies B_block (a, bc) -> C_block (bc) at rows bc*2..bc*2+1

    for a in range(2):
        for bc in range(3):
            for s in range(7):
                # U: A_block a, local (i, j)
                for i in range(2):
                    for j in range(2):
                        val = strassen_U[i * 2 + j][s]
                        if val != 0:
                            u_idx = i * m + (a * 2 + j)
                            U[u_idx][col] += val

                # V: B block (a, bc), local (k, l)
                for k in range(2):
                    for l in range(2):
                        val = strassen_V[k * 2 + l][s]
                        if val != 0:
                            v_idx = (a * 2 + k) * p + (bc * 2 + l)
                            V[v_idx][col] += val

                # W: C block bc, local (k, i)
                for k in range(2):
                    for i in range(2):
                        val = strassen_W[k * 2 + i][s]
                        if val != 0:
                            w_idx = (bc * 2 + k) * n + i
                            W[w_idx][col] += val

                col += 1

    return {"U": U, "V": V, "W": W}
