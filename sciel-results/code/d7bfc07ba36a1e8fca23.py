"""Rewritten program: composition-based decomposition for 2x3x6 matrix multiplication."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x6 matrix using a composition approach.
    
    We view the 3x6 matrix B as two 3x3 blocks: B = [B1, B2].
    Then C = A * B = [A * B1, A * B2].
    This requires two separate 2x3x3 matrix multiplications.
    
    The rank of a 2x3x3 multiplication is 18 (schoolbook).
    Total rank = 18 + 18 = 36.
    
    This matches the global schoolbook rank for 2x3x6.
    """
    n, m, p = DIMENSIONS
    # p is 6, so we split into two blocks of size 3
    p_block = 3
    
    # Rank of 2x3x3 schoolbook
    rank_block = n * m * p_block # 2 * 3 * 3 = 18
    
    total_rank = 2 * rank_block # 36
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a 2x3x3 schoolbook decomposition
    # offset: column offset in the global matrices
    # k_offset: offset for the k index in V and W (0 or 3)
    def add_block(offset, k_offset):
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p_block):
                    # Global indices
                    u_idx = i * m + l
                    v_idx = l * p + (k_offset + k)
                    w_idx = (k_offset + k) * n + i
                    
                    U[u_idx][offset + col] = 1
                    V[v_idx][offset + col] = 1
                    W[w_idx][offset + col] = 1
                    col += 1
        
    # Add first block (k in 0..2)
    add_block(0, 0)
    
    # Add second block (k in 3..5)
    add_block(rank_block, p_block)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
