import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # We split B into p//2 blocks of size 2x2
    num_blocks = p // 2
    # Rank of 2x2x2 is 7
    rank_per_block = 7
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11) b00
    # M3 = a00 (b10 - b11)
    # M4 = a11 (b01 + b11)
    # M5 = (a00 + a01) b11
    # M6 = (a01 - a00)(b00 + b10)
    # M7 = (a10 - a11)(b01 + b11)
    
    # U vectors (for A)
    # a00 is at index 0, a01 at 1, a10 at 2, a11 at 3
    u_terms = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 1, -1, 0],# M6
        [0, 0, 1, -1] # M7
    ]
    
    # V vectors (for B)
    # b00 is at index 0, b01 at 1, b10 at 2, b11 at 3
    v_terms = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, -1, 1, 0],# M3
        [0, 1, 0, 1], # M4
        [0, 0, 0, 1], # M5
        [1, 0, 1, 0], # M6
        [0, 1, 0, 1]  # M7
    ]
    
    # W vectors (for C)
    # c00 is at index 0, c01 at 1, c10 at 2, c11 at 3
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M7
    # c11 = M1 - M2 + M3 + M6
    
    w_terms = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 0, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [-1, 0, 1, 0] # M7
    ]
    
    for k in range(num_blocks):
        for r in range(rank_per_block):
            col = k * rank_per_block + r
            
            # Fill U
            for i in range(n * m):
                U[i][col] = u_terms[r][i]
                
            # Fill V
            # B is 2x14. Block k is columns 2k, 2k+1
            # b00 is at (0, 2k), b01 at (0, 2k+1), b10 at (1, 2k), b11 at (1, 2k+1)
            # V index is j*p + k_v
            # j is row of B, k_v is col of B
            # b00: j=0, k_v=2k -> index 0*14 + 2k = 2k
            # b01: j=0, k_v=2k+1 -> index 2k+1
            # b10: j=1, k_v=2k -> index 14 + 2k
            # b11: j=1, k_v=2k+1 -> index 14 + 2k + 1
            
            v_indices = [2*k, 2*k+1, 14+2*k, 14+2*k+1]
            for i in range(4):
                V[v_indices[i]][col] = v_terms[r][i]
                
            # Fill W
            # C is 2x14. Block k is columns 2k, 2k+1
            # c00 is at (0, 2k), c01 at (0, 2k+1), c10 at (1, 2k), c11 at (1, 2k+1)
            # W index is k_v*n + i
            # k_v is col of C, i is row of C
            # c00: k_v=2k, i=0 -> index 2k*2 + 0 = 4k
            # c01: k_v=2k+1, i=0 -> index (2k+1)*2 + 0 = 4k+2
            # c10: k_v=2k, i=1 -> index 2k*2 + 1 = 4k+1
            # c11: k_v=2k+1, i=1 -> index (2k+1)*2 + 1 = 4k+3
            
            w_indices = [4*k, 4*k+2, 4*k+1, 4*k+3]
            for i in range(4):
                W[w_indices[i]][col] = w_terms[r][i]
                
    return {"U": U, "V": V, "W": W}
