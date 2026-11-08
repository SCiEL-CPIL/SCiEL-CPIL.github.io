"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strassen 2x2x2 rank-7 factors
    # U: 4x7, V: 4x7, W: 4x7
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1],
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # We decompose 4x4x6 into 2x2x2 blocks.
    # n=4, m=4, p=6.
    # Blocks: ni=2, mj=2, pk=3.
    # Total blocks: 12.
    # Rank per block: 7.
    # Total rank: 84.
    
    ni, mj, pk = 2, 2, 3
    block_size = 2
    rank_per_block = 7
    total_rank = ni * mj * pk * rank_per_block
    
    # Initialize matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for bi in range(ni):
        for bj in range(mj):
            for bk in range(pk):
                for c in range(rank_per_block):
                    col = col_offset + c
                    
                    # Fill U (n*m x rank)
                    # Row index: i * m + j
                    # i in [bi*2, bi*2+2), j in [bj*2, bj*2+2)
                    for di in range(2):
                        for dj in range(2):
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            U[i * m + j][col] = S_U[di * 2 + dj][c]
                            
                    # Fill V (m*p x rank)
                    # Row index: j * p + k
                    # j in [bj*2, bj*2+2), k in [bk*2, bk*2+2)
                    for dj in range(2):
                        for dk in range(2):
                            j = bj * 2 + dj
                            k = bk * 2 + dk
                            V[j * p + k][col] = S_V[dj * 2 + dk][c]
                            
                    # Fill W (p*n x rank)
                    # Row index: k * n + i
                    # k in [bk*2, bk*2+2), i in [bi*2, bi*2+2)
                    for dk in range(2):
                        for di in range(2):
                            k = bk * 2 + dk
                            i = bi * 2 + di
                            W[k * n + i][col] = S_W[dk * 2 + di][c]
                
                col_offset += rank_per_block
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
