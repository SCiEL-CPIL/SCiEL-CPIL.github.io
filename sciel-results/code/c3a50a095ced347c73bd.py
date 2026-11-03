import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size
    
    rank = n_blocks**3 * 7
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)
        
    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)
        
    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen steps defined as (u_coords, v_coords, w_coords)
    # Each step is a list of (idx, val) for U, V, W
    steps = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                for u_defs, v_defs, w_defs in steps:
                    for r, c, val in u_defs:
                        U[u_idx(ib, kb, r, c), col] += val
                    for r, c, val in v_defs:
                        V[v_idx(kb, jb, r, c), col] += val
                    for r, c, val in w_defs:
                        W[w_idx(ib, jb, r, c), col] += val
                    col += 1
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
