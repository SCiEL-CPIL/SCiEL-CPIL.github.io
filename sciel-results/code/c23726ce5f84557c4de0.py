import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    deadline = start_time + 110  # Safety margin

    # Block structure: 6x6 = 2x2 blocks of 3x3
    # Outer: 2x2 matrix multiplication (Strassen rank 7)
    # Inner: 3x3 matrix multiplication (Naive rank 27)
    # Total Rank: 7 * 27 = 189
    
    block_size = 3
    num_blocks = 2
    dim = n  # 6
    
    rank_inner = block_size ** 3  # 27
    rank_outer = 7  # Strassen rank for 2x2
    total_rank = rank_inner * rank_outer  # 189
    
    # Initialize matrices
    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)
    
    # Strassen coefficients for 2x2 multiplication
    # Each step is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # Coeffs are lists of (row, col, coeff)
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    bs = block_size
    
    # Iterate over Strassen steps
    for x_list, y_list, c_list in strassen_2x2:
        # Iterate over inner 3x3 block indices
        # r: row in A block, s: col in A block (and row in B block)
        # c: col in B block (and col in C block)
        # Wait, the inner loop in the previous code was:
        # for r in range(block_size):
        #     for s in range(block_size):
        #         for c in range(block_size):
        # This corresponds to the 27 terms of the 3x3 multiplication.
        # The indices in the 3x3 block are (r,s) for A, (s,c) for B, (r,c) for C.
        
        for r in range(bs):
            for s in range(bs):
                for c in range(bs):
                    # Update U
                    for (ib, jb, x_coeff) in x_list:
                        # Global index for A:
                        # Row: ib * bs + r
                        # Col: jb * bs + s
                        u_idx = (ib * bs + r) * dim + (jb * bs + s)
                        U[u_idx, col] += x_coeff
                    
                    # Update V
                    for (jb, kb, y_coeff) in y_list:
                        # Global index for B:
                        # Row: jb * bs + s
                        # Col: kb * bs + c
                        v_idx = (jb * bs + s) * dim + (kb * bs + c)
                        V[v_idx, col] += y_coeff
                    
                    # Update W
                    for (ib, kb, c_coeff) in c_list:
                        # Global index for C:
                        # Row: ib * bs + r
                        # Col: kb * bs + c
                        # Trace convention: W[k*n+i, r]
                        # k = col_C, i = row_C
                        w_idx = (kb * bs + c) * dim + (ib * bs + r)
                        W[w_idx, col] += c_coeff
                    
                    col += 1
                    
                    # Periodic time check
                    if col % 189 == 0 and time.time() > deadline:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
