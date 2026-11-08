"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Use 2x2x2 Strassen rank-7 decomposition for each block
    # For 4x4 x 4x6: split into 2x2 blocks in n and m, and 2x2 blocks in p (3 blocks)
    
    # 2x2x2 Strassen factors (rank 7)
    # U: 4x7, rows (i,j) for i,j in {0,1}
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ]
    # V: 4x7, rows (j,k) for j,k in {0,1}
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ]
    # W: 4x7, rows (k,i) for k,i in {0,1}
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ]
    
    # Block structure: n=4 (2 blocks of 2), m=4 (2 blocks of 2), p=6 (3 blocks of 2)
    n_blocks = n // 2  # 2
    m_blocks = m // 2  # 2
    p_blocks = p // 2  # 3
    block_size = 2
    
    # Total rank = n_blocks * m_blocks * p_blocks * 7 = 2 * 2 * 3 * 7 = 84
    total_rank = n_blocks * m_blocks * p_blocks * 7
    
    # Initialize factor matrices
    U = [[0] * total_rank for _ in range(n * m)]  # (n*m) x total_rank = 16 x 84
    V = [[0] * total_rank for _ in range(m * p)]  # (m*p) x total_rank = 24 x 84
    W = [[0] * total_rank for _ in range(p * n)]  # (p*n) x total_rank = 24 x 84
    
    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                # For each of the 7 Strassen products
                for c in range(7):
                    col = col_offset + c
                    # U: rows (i,j) for i in [bi*2, bi*2+2), j in [bj*2, bj*2+2)
                    for di in range(block_size):
                        for dj in range(block_size):
                            i = bi * block_size + di
                            j = bj * block_size + dj
                            U[i * m + j][col] = S_U[di * block_size + dj][c]
                    
                    # V: rows (j,k) for j in [bj*2, bj*2+2), k in [bk*2, bk*2+2)
                    for dj in range(block_size):
                        for dk in range(block_size):
                            j = bj * block_size + dj
                            k = bk * block_size + dk
                            V[j * p + k][col] = S_V[dj * block_size + dk][c]
                    
                    # W: rows (k,i) for k in [bk*2, bk*2+2), i in [bi*2, bi*2+2)
                    for dk in range(block_size):
                        for di in range(block_size):
                            k = bk * block_size + dk
                            i = bi * block_size + di
                            W[k * n + i][col] = S_W[dk * block_size + di][c]
                
                col_offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
