import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen 2x2 * 2x2 decomposition (rank 7)
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ], dtype=float)
    
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ], dtype=float)
    
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ], dtype=float)
    
    # Build the 56-rank Strassen solution
    # We have 8 independent 2x2x2 multiplications
    # A is split into A1 (cols 0,1) and A2 (cols 2,3)
    # B is split into B1 (rows 0,1) and B2 (rows 2,3)
    # B1 is split into 4 blocks of 2 cols: B1_0, B1_1, B1_2, B1_3
    # B2 is split into 4 blocks of 2 cols: B2_0, B2_1, B2_2, B2_3
    # Products: A1*B1_j and A2*B2_j for j=0..3
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # A1 mapping: local (i, j) -> global i*4 + j
    # A1 is cols 0,1.
    # Local index la: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global index: i*4 + j
    a1_map = [0, 1, 4, 5]
    
    # A2 mapping: local (i, j) -> global i*4 + 2 + j
    # A2 is cols 2,3.
    a2_map = [2, 3, 6, 7]
    
    # B1 mapping: local (j, k) -> global j*8 + 2*block + k
    # B1 is rows 0,1.
    # For block j_b (0..3), cols are 2*j_b, 2*j_b+1.
    # Local index lb: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global index: row*8 + col
    
    # W mapping: local (k, i) -> global k*2 + i ?
    # W index is k*n + i = 2k + i.
    # k is col of C. i is row of C.
    # For block j_b, k is 2*j_b + k_local.
    # So W index = (2*j_b + k_local)*2 + i = 4*j_b + 2*k_local + i.
    
    def add_strassen_block(U, V, W, base_rank, a_map, b_row_offset, b_col_offset, w_base):
        """
        Add a Strassen block.
        a_map: list of 4 global U indices for A block.
        b_row_offset: row offset for B block (0 or 2).
        b_col_offset: col offset for B block (0, 2, 4, 6).
        w_base: base index for W block.
        """
        for l in range(7):
            r = base_rank + l
            # U
            for la in range(4):
                if U_local[la, l] != 0:
                    U[a_map[la], r] += U_local[la, l]
            # V
            # B block is 2x2. Rows are b_row_offset, b_row_offset+1.
            # Cols are b_col_offset, b_col_offset+1.
            # Local V index lb: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global V index: row*8 + col
            v_indices = [
                (b_row_offset) * 8 + b_col_offset,
                (b_row_offset) * 8 + b_col_offset + 1,
                (b_row_offset + 1) * 8 + b_col_offset,
                (b_row_offset + 1) * 8 + b_col_offset + 1
            ]
            for lb in range(4):
                if V_local[lb, l] != 0:
                    V[v_indices[lb], r] += V_local[lb, l]
            # W
            # W index is k*n + i.
            # k is col of C. i is row of C.
            # For this block, k = b_col_offset + k_local.
            # i = i_local.
            # Local W index lc: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Corresponds to (k_local=0, i=0), (k_local=0, i=1), (k_local=1, i=0), (k_local=1, i=1)
            # Global W index: (b_col_offset + k_local)*n + i
            w_indices = [
                (b_col_offset + 0) * n + 0,
                (b_col_offset + 0) * n + 1,
                (b_col_offset + 1) * n + 0,
                (b_col_offset + 1) * n + 1
            ]
            for lc in range(4):
                if W_local[lc, l] != 0:
                    W[w_indices[lc], r] += W_local[lc, l]

    # Add 4 blocks for A1 * B1_j
    for j_b in range(4):
        base_r = j_b * 7
        b_row_offset = 0
        b_col_offset = j_b * 2
        add_strassen_block(U, V, W, base_r, a1_map, b_row_offset, b_col_offset, 0)
        
    # Add 4 blocks for A2 * B2_j
    for j_b in range(4):
        base_r = 28 + j_b * 7
        b_row_offset = 2
        b_col_offset = j_b * 2
        add_strassen_block(U, V, W, base_r, a2_map, b_row_offset, b_col_offset, 0)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
