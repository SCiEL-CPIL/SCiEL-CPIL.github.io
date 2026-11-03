import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    # Using trivial 3x3 decomposition (rank 27) inside Strassen (7 steps)
    # Total rank 189
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    gamma = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for i in range(7):
        for r in range(3):
            for i_sub in range(3):
                for s in range(3):
                    # Check time
                    if time.time() > deadline - 5.0:
                        return {"U": U, "V": V, "W": W}
                    
                    # Column index in the decomposition
                    # 27 terms per Strassen step
                    k = r * 9 + i_sub * 3 + s
                    c = i * 27 + k
                    
                    for b in range(4):
                        # Calculate global row and col indices for standard row-major flattening
                        # Block b: 0(TL), 1(TR), 2(BL), 3(BR)
                        row_block = b // 2
                        col_block = b % 2
                        
                        # U: A[r, i_sub] in block b
                        # Global Row: 3*row_block + r
                        # Global Col: 3*col_block + i_sub
                        u_idx = (3 * row_block + r) * 6 + (3 * col_block + i_sub)
                        
                        # V: B[i_sub, s] in block b
                        # Global Row: 3*row_block + i_sub
                        # Global Col: 3*col_block + s
                        v_idx = (3 * row_block + i_sub) * 6 + (3 * col_block + s)
                        
                        # W: C[r, s] in block b
                        # Global Row: 3*row_block + r
                        # Global Col: 3*col_block + s
                        w_idx = (3 * row_block + r) * 6 + (3 * col_block + s)
                        
                        if alpha[i][b] != 0:
                            U[u_idx, c] = alpha[i][b]
                            
                        if beta[i][b] != 0:
                            V[v_idx, c] = beta[i][b]
                            
                        if gamma[i][b] != 0:
                            W[w_idx, c] = gamma[i][b]
                            
    return {"U": U, "V": V, "W": W}
