"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    block_sizes = [2, 2, 1]
    
    def block_offsets(sizes):
        offsets = []
        off = 0
        for s in sizes:
            offsets.append(off)
            off += s
        return offsets
    
    off_i = block_offsets(block_sizes)
    off_j = block_offsets(block_sizes)
    off_k = block_offsets(block_sizes)
    
    U_entries = []
    V_entries = []
    W_entries = []
    col = 0
    
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                si = block_sizes[bi]
                sj = block_sizes[bj]
                sk = block_sizes[bk]
                
                if si == 2 and sj == 2 and sk == 2:
                    u_forms = [
                        [[1, 0], [0, 1]],
                        [[0, 0], [1, 1]],
                        [[1, 0], [0, 0]],
                        [[0, 0], [0, 1]],
                        [[1, 1], [0, 0]],
                        [[-1, 0], [1, 0]],
                        [[0, 1], [0, -1]],
                    ]
                    v_forms = [
                        [[1, 0], [0, 1]],
                        [[1, 0], [0, 0]],
                        [[0, 1], [0, -1]],
                        [[0, -1], [1, 0]],
                        [[0, 0], [0, 1]],
                        [[1, 1], [0, 0]],
                        [[0, 1], [1, 1]],
                    ]
                    w_forms = [
                        [[1, 0], [0, 1]],
                        [[0, 0], [1, 0]],
                        [[0, 1], [0, 0]],
                        [[1, 0], [1, 0]],
                        [[-1, 1], [0, 0]],
                        [[0, 0], [0, 1]],
                        [[1, 0], [0, 0]],
                    ]
                    
                    for t in range(7):
                        for li in range(2):
                            for lj in range(2):
                                val = u_forms[t][li][lj]
                                if val != 0:
                                    gr = (off_i[bi] + li) * m + (off_j[bj] + lj)
                                    U_entries.append((gr, val, col))
                        for lj in range(2):
                            for lk in range(2):
                                val = v_forms[t][lj][lk]
                                if val != 0:
                                    gr = (off_j[bj] + lj) * p + (off_k[bk] + lk)
                                    V_entries.append((gr, val, col))
                        for lk in range(2):
                            for li in range(2):
                                val = w_forms[t][lk][li]
                                if val != 0:
                                    gr = (off_k[bk] + lk) * n + (off_i[bi] + li)
                                    W_entries.append((gr, val, col))
                        col += 1
                else:
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                gr_u = (off_i[bi] + li) * m + (off_j[bj] + lj)
                                U_entries.append((gr_u, 1, col))
                                gr_v = (off_j[bj] + lj) * p + (off_k[bk] + lk)
                                V_entries.append((gr_v, 1, col))
                                gr_w = (off_k[bk] + lk) * n + (off_i[bi] + li)
                                W_entries.append((gr_w, 1, col))
                                col += 1
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for row, val, c in U_entries:
        U[row][c] = val
    for row, val, c in V_entries:
        V[row][c] = val
    for row, val, c in W_entries:
        W[row][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
