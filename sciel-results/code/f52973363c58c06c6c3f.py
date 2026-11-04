"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Rank-14 construction for 2x3x10
    # Split into two 2x3x5 blocks, each with rank 7
    # For 2x3x5: use Strassen on the 2x2x2 submatrix (cols 0,1) with rank 7
    # and handle cols 2,3,4 with 3 additional terms -> rank 10, not 7
    # 
    # Alternative: use a known rank-7 decomposition for 2x3x5
    # Let me use the approach: for 2x3x5, the rank is at least 2*3*5/2 = 15? No.
    # 
    # Let me just implement schoolbook rank 30 correctly to ensure validity,
    # then the evaluator will score it.
    
    n, m, p = DIMENSIONS
    block_p = 5
    num_blocks = p // block_p
    total_rank = n * m * block_p * num_blocks  # 30
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(block_p):
                    u_idx = i * m + j
                    v_idx = j * p + (b * block_p + k)
                    w_idx = (b * block_p + k) * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
