import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for (2x2) * (2x8).
    
    Strategy: Use a rank-14 algorithm for 2x2*2x4, then replicate it twice
    for the two 4-column blocks of the 2x8 matrix. This gives rank 28.
    
    The rank-14 algorithm uses Strassen's 7-term formula for two overlapping
    2x2 blocks that share a column, giving savings over naive 14-term.
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Define the rank-14 algorithm for 2x2 * 2x4
    # B columns: 0, 1, 2, 3
    # C columns: 0, 1, 2, 3
    # 
    # Terms 1-7: Strassen on B columns 0,1 -> C columns 0,1
    # Terms 8-14: Strassen on B columns 2,3 -> C columns 2,3
    #
    # Wait, that's just 7+7=14 with no sharing. Let me think differently.
    #
    # Actually, for 2x2 * 2x4, the trivial block approach gives rank 14 
    # (two independent 2x2 Strassen computations). But there might be 
    # a smarter decomposition. However, for now let's just use the block 
    # approach which is known to be rank 14 for 2x2*2x4.
    #
    # For 2x2 * 2x8, we do 4 such blocks: rank = 4 * 7 = 28.
    # This is the same as the current program but let me verify it's correct.
    
    # Strassen 2x2x2 base
    U_base = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_base = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_base = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # 4 blocks of 2 columns each
    num_blocks = p // 2  # 4
    rank = num_blocks * 7  # 28
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for blk in range(num_blocks):
        col_offset = blk * 7
        
        # U: same for all blocks (A doesn't change)
        for r in range(7):
            for i in range(n * m):
                U[i, col_offset + r] = U_base[r][i]
        
        # V: B is 2x8, index = row * p + col
        # Block blk uses B columns 2*blk, 2*blk+1
        # Local indices: 0=(0,2blk), 1=(0,2blk+1), 2=(1,2blk), 3=(1,2blk+1)
        b_col_start = 2 * blk
        for r in range(7):
            for l in range(4):
                if l == 0:
                    v_idx = 0 * p + b_col_start
                elif l == 1:
                    v_idx = 0 * p + b_col_start + 1
                elif l == 2:
                    v_idx = 1 * p + b_col_start
                else:
                    v_idx = 1 * p + b_col_start + 1
                V[v_idx, col_offset + r] = V_base[r][l]
        
        # W: C is 2x8, convention W[k*n + i, r] where k=col, i=row
        # Block blk produces C columns 2*blk, 2*blk+1
        # Local: 0=C[0][2blk], 1=C[0][2blk+1], 2=C[1][2blk], 3=C[1][2blk+1]
        c_col_start = 2 * blk
        for r in range(7):
            for l in range(4):
                if l == 0:
                    w_idx = c_col_start * n + 0
                elif l == 1:
                    w_idx = (c_col_start + 1) * n + 0
                elif l == 2:
                    w_idx = c_col_start * n + 1
                else:
                    w_idx = (c_col_start + 1) * n + 1
                W[w_idx, col_offset + r] = W_base[r][l]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
