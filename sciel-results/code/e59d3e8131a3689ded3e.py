import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Two blocks for k in 0..3 and 4..7
    for k_offset in [0, 4]:
        # 3 terms for i=0, 4 terms for i=1, 4 terms for i=2?
        # No, 4 terms for i=0 (j=0..3), 4 terms for i=1, 4 terms for i=2.
        # Total 12 per block.
        for i in range(n):
            for j in range(m):
                # u
                U[i * m + j, col] = 1
                # v
                # index in V is j * p + k
                # for this block, k = k_offset + k_local
                # we need v to be 1 for all k_local in 0..3
                # so v[j * p + k_offset + k_local] = 1
                for k_local in range(4):
                    V[j * p + k_offset + k_local, col] = 1
                # w
                # index in W is k * n + i
                # w[k * n + i] = 1 for all k_local in 0..3
                for k_local in range(4):
                    W[(k_offset + k_local) * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
