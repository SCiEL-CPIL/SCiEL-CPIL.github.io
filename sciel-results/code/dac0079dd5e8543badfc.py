import time
import numpy as np

def construct():
    """
    Rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Block strategy: Strassen(4x4x4) for B[:,0:4] and B[:,4:8], schoolbook for B[:,8].
    """
    n, m, p = 4, 4, 9

    # Strassen 2x2x2: 7 rank-1 terms
    # (u_mask, v_mask, w_mask) where masks are dicts {(i,j): coeff}
    S2 = [
        ({(0,0):1,(1,1):1},  {(0,0):1,(1,1):1},  {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1},  {(0,0):1},          {(1,0):1,(1,1):-1}),
        ({(0,0):1},          {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1},          {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1},  {(1,1):1},          {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1},  {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1},  {(0,0):1}),
    ]

    # 4x4x4 Strassen: 7 combinations of 2x2 blocks
    # (a_blocks, b_blocks, c_blocks)
    # Each block entry is (block_row, block_col, coeff)
    S4 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)],          [(1,0,1),(1,1,-1)]),
        ([(0,0,1)],          [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)],          [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)],  [(1,1,1)],          [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)],  [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)],  [(0,0,1)]),
    ]

    U_list = []
    V_list = []
    W_list = []

    def emit(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    def strassen_4x4(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off):
        """Emit 49 rank-1 terms for a 4x4x4 sub-problem."""
        for a_blocks, b_blocks, c_blocks in S4:
            for (u_mask, v_mask, w_mask) in S2:
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)

                for (li, lj), cu in u_mask.items():
                    for (br, bc, cb) in a_blocks:
                        gi = a_row_off + br * 2 + li
                        gj = a_col_off + bc * 2 + lj
                        u[gi * m + gj] += cu * cb

                for (lj, lk), cv in v_mask.items():
                    for (br, bc, cb) in b_blocks:
                        gj = b_row_off + br * 2 + lj
                        gk = b_col_off + bc * 2 + lk
                        v[gj * p + gk] += cv * cb

                for (li, lk), cw in w_mask.items():
                    for (br, bc, cb) in c_blocks:
                        gi = c_row_off + br * 2 + li
                        gk = c_col_off + bc * 2 + lk
                        w[gk * n + gi] += cw * cb

                emit(u, v, w)

    # B columns 0-3: Strassen 4x4x4
    strassen_4x4(0, 0, 0, 0, 0, 0)
    # B columns 4-7: Strassen 4x4x4
    strassen_4x4(0, 0, 0, 4, 0, 4)
    # B column 8: schoolbook 4x1x4
    for i in range(n):
        for j in range(m):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[i * m + j] = 1
            v[j * p + 8] = 1
            w[8 * n + i] = 1
            emit(u, v, w)

    r = len(U_list)
    # Transpose to get (rows, rank) format
    U = [[U_list[t][i] for t in range(r)] for i in range(n * m)]
    V = [[V_list[t][i] for t in range(r)] for i in range(m * p)]
    W = [[W_list[t][i] for t in range(r)] for i in range(p * n)]

    return {"U": U, "V": V, "W": W}
