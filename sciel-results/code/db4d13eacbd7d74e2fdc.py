"""Strassen-based 2x5 x 5x6 matrix multiplication decomposition."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    n, m, p = 2, 5, 6
    
    # Target rank: 6 Strassen blocks (7 each) + 12 remainder terms = 54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(start_rank, i_off, j_off, k_off):
        # i_off: 0
        # j_off: 0 or 2
        # k_off: 0, 2, or 4
        
        terms = [
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # M3
            ({(0,0): 1}, {(1,0): -1, (1,1): 1}, {(0,1): 1}),
            # M4
            ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1}),
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,1): 1}),
            # M6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, {(1,0): 1}),
            # M7
            ({(0,1): 1, (1,1): 1}, {(0,1): 1, (1,1): 1}, {(0,1): -1, (1,1): 1}),
        ]
        
        for t_idx, (u_c, v_c, w_c) in enumerate(terms):
            r = start_rank + t_idx
            for (li, lj), val in u_c.items():
                gi = i_off + li
                gj = j_off + lj
                U[gi * m + gj][r] = val
            for (lj, lk), val in v_c.items():
                gj = j_off + lj
                gk = k_off + lk
                V[gj * p + gk][r] = val
            for (lk, li), val in w_c.items():
                gk = k_off + lk
                gi = i_off + li
                W[gk * n + gi][r] = val

    # 6 Strassen blocks
    # A is 2x5, B is 5x6
    # A cols: 0,1 (block 1), 2,3 (block 2), 4 (remainder)
    # B cols: 0,1 (block 1), 2,3 (block 2), 4,5 (block 3)
    
    # Block 1: A[:, 0:2] * B[0:2, :]
    # This covers B cols 0,1 and 2,3 and 4,5? No.
    # We split B into 2x2 blocks for the Strassen part.
    # B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
    # B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6]
    
    # Strassen Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    add_strassen_block(0, 0, 0, 0)
    
    # Strassen Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    add_strassen_block(7, 0, 0, 2)
    
    # Strassen Block 3: A[:, 0:2] * B[0:2, 4:6] -> C[:, 4:6]
    add_strassen_block(14, 0, 0, 4)
    
    # Strassen Block 4: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    add_strassen_block(21, 0, 2, 0)
    
    # Strassen Block 5: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    add_strassen_block(28, 0, 2, 2)
    
    # Strassen Block 6: A[:, 2:4] * B[2:4, 4:6] -> C[:, 4:6]
    add_strassen_block(35, 0, 2, 4)
    
    # Remainder: A[:, 4:5] * B[4:5, :]
    # 12 terms
    r_start = 42
    for i in range(2):
        for k in range(6):
            r = r_start + i * 6 + k
            U[i * 5 + 4][r] = 1
            V[4 * 6 + k][r] = 1
            W[k * 2 + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
