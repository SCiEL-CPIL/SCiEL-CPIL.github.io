"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for M_{2,2,2}
    # Rank 7
    U_s = [
        [1, 0, 0, 0, 0, 0, 0],  # i=0, j=0
        [0, 1, 0, 1, 1, 0, 0],  # i=0, j=1
        [0, 0, 1, 0, 0, 1, 1],  # i=1, j=0
        [0, 0, 0, 1, 1, 0, 1],  # i=1, j=1
    ]
    V_s = [
        [1, 1, 0, 0, 0, 0, 0],  # j=0, k=0
        [0, 0, 1, 1, 1, 0, 0],  # j=0, k=1
        [1, 1, 0, 0, 0, 0, 0],  # j=1, k=0
        [0, 0, 1, 1, 1, 0, 0],  # j=1, k=1
    ]
    W_s = [
        [1, 0, 0, 0, 0, 0, 0],  # k=0, i=0
        [0, 1, 1, 1, 0, 0, 0],  # k=0, i=1
        [0, 0, 1, 0, 1, 1, 0],  # k=1, i=0
        [0, 0, 0, 1, 1, 0, 1],  # k=1, i=1
    ]
    
    # Schoolbook decomposition for M_{2,2,1}
    # Rank 4
    U_sc = [
        [1, 0, 0, 0],  # i=0, j=0
        [0, 1, 0, 0],  # i=0, j=1
        [0, 0, 1, 0],  # i=1, j=0
        [0, 0, 0, 1],  # i=1, j=1
    ]
    V_sc = [
        [1, 0, 0, 0],  # j=0, k=0
        [0, 1, 0, 0],  # j=1, k=0
    ]
    W_sc = [
        [1, 0, 0, 0],  # k=0, i=0
        [0, 0, 1, 0],  # k=0, i=1
    ]
    
    # Combine: 7 blocks of Strassen (covering k=0..13) + 1 block of Schoolbook (covering k=14)
    # Total rank = 7 * 7 + 4 = 53
    num_blocks_strassen = 7
    num_blocks_schoolbook = 1
    
    total_rank = num_blocks_strassen * 7 + num_blocks_schoolbook * 4
    
    # Initialize U (n*m x rank), V (m*p x rank), W (p*n x rank)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Place Strassen blocks
    for block in range(num_blocks_strassen):
        k_start = block * 2
        for r in range(7):
            # U: same for all blocks
            for i_m in range(4):
                U[i_m][col_offset] = U_s[i_m][r]
            
            # V: map k indices to block's k range
            for j_p in range(4):
                j = j_p // 2
                k_local = j_p % 2
                V[j * p + k_start + k_local][col_offset] = V_s[j_p][r]
            
            # W: map k indices to block's k range
            for k_n in range(4):
                k_local = k_n // 2
                i = k_n % 2
                W[(k_start + k_local) * n + i][col_offset] = W_s[k_n][r]
            
            col_offset += 1
    
    # Place Schoolbook block for k=14
    k_start = 14
    for r in range(4):
        # U
        for i_m in range(4):
            U[i_m][col_offset] = U_sc[i_m][r]
        
        # V
        for j_p in range(2):
            j = j_p // 2
            k_local = j_p % 2
            V[j * p + k_start + k_local][col_offset] = V_sc[j_p][r]
        
        # W
        for k_n in range(2):
            k_local = k_n // 2
            i = k_n % 2
            W[(k_start + k_local) * n + i][col_offset] = W_sc[k_n][r]
        
        col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
