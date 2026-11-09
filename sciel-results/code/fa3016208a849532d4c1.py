import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition matrices
    # Row/Col index in 2x2 blocks: 0:00, 1:01, 2:10, 3:11 (row-major)
    # W2 row index corresponds to output element index k*2+i (col*2+row)
    # Index 0: C_00, Index 1: C_10, Index 2: C_01, Index 3: C_11
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # 00
        [0, 0, 0, 0, 1, 0, 1],   # 01
        [0, 1, 0, 0, 0, 1, 0],   # 10
        [1, 1, 0, 1, 0, 0, -1]   # 11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # 00
        [0, 0, 1, 0, 0, 1, 0],   # 01
        [0, 0, 0, 1, 0, 0, 1],   # 10
        [1, 0, -1, 0, 1, 0, 1]   # 11
    ]
    
    # W2 rows ordered by k*2+i (col*2+row):
    # 0: C_00 (k=0, i=0)
    # 1: C_10 (k=0, i=1)
    # 2: C_01 (k=1, i=0)
    # 3: C_11 (k=1, i=1)
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],  # 0: C_00
        [0, 1, 0, 1, 0, 0, 0],   # 1: C_10
        [0, 0, 1, 0, 1, 0, 0],   # 2: C_01
        [1, -1, 1, 0, 0, 1, 0]   # 3: C_11
    ]

    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U and V
    for I in range(2):
        for J in range(2):
            block_idx = I * 2 + J
            for i in range(2):
                for j in range(2):
                    inner_idx = i * 2 + j
                    u_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    v_idx = (I * 2 + i) * 4 + (J * 2 + j)
                    
                    for r in range(7):
                        ub = U2[block_idx][r]
                        vb = V2[block_idx][r]
                        if ub == 0 and vb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            ui = U2[inner_idx][s]
                            vi = V2[inner_idx][s]
                            if ui == 0 and vi == 0:
                                continue
                            U[u_idx][c] = ub * ui
                            V[v_idx][c] = vb * vi

    # Fill W
    for I in range(2):
        for J in range(2):
            # W2 block index is J*2 + I (col*2+row)
            block_idx = J * 2 + I
            for i in range(2):
                for j in range(2):
                    # W2 inner index is j*2 + i (col*2+row)
                    inner_idx = j * 2 + i
                    w_idx = (J * 2 + j) * 4 + (I * 2 + i)
                    
                    for r in range(7):
                        wb = W2[block_idx][r]
                        if wb == 0:
                            continue
                        for s in range(7):
                            c = r * 7 + s
                            wi = W2[inner_idx][s]
                            if wi == 0:
                                continue
                            W[w_idx][c] = wb * wi

    # Prune any zero columns
    keep = [c for c in range(rank) if not (
        all(U[r][c] == 0 for r in range(n * m)) or 
        all(V[r][c] == 0 for r in range(m * p)) or 
        all(W[r][c] == 0 for r in range(p * n))
    )]
    if len(keep) < rank:
        U = [[row[c] for c in keep] for row in U]
        V = [[row[c] for c in keep] for row in V]
        W = [[row[c] for c in keep] for row in W]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
