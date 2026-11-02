"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 5 = 3 + 1 + 1 block decomposition with schoolbook for each block
    # Total rank: 27 + 3 + 3 + 3 + 1 + 1 + 3 + 1 + 1 = 43
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Block offsets for 5 = 3 + 1 + 1
    row_starts = [0, 3, 4]
    col_starts = [0, 3, 4]
    out_starts = [0, 3, 4]
    row_sizes = [3, 1, 1]
    col_sizes = [3, 1, 1]
    out_sizes = [3, 1, 1]
    
    def add_schoolbook_block(bi, bj, bk):
        rs = row_starts[bi]
        cs = col_starts[bj]
        os_ = out_starts[bk]
        rsz = row_sizes[bi]
        csz = col_sizes[bj]
        osz = out_sizes[bk]
        
        for i in range(rsz):
            for j in range(csz):
                for k in range(osz):
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    u[(rs + i) * m + (cs + j)] = 1
                    v[(cs + j) * p + (os_ + k)] = 1
                    w[(os_ + k) * n + (rs + i)] = 1
                    add_term(u, v, w)
    
    # Add all 9 blocks
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                # Block (bi, bj) of A, block (bj, bk) of B, block (bi, bk) of C
                add_schoolbook_block(bi, bj, bk)
    
    total_rank = len(U_cols)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
