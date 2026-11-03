"""Rank-98 tensor decomposition for 4x4 * 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Inner Strassen 2x2x2 products
    inner_products = [
        {'A': {(0,0):1, (1,1):1}, 'B': {(0,0):1, (1,1):1}, 'C': {(0,0):1, (1,1):1}},
        {'A': {(1,0):1, (1,1):1}, 'B': {(0,0):1}, 'C': {(1,0):1, (1,1):-1}},
        {'A': {(0,0):1}, 'B': {(1,0):1, (1,1):-1}, 'C': {(0,1):1, (1,1):1}},
        {'A': {(1,1):1}, 'B': {(0,1):1, (0,0):-1}, 'C': {(0,0):1, (1,0):1}},
        {'A': {(0,0):1, (0,1):1}, 'B': {(1,1):1}, 'C': {(0,0):-1, (0,1):1}},
        {'A': {(1,0):1, (0,0):-1}, 'B': {(0,0):1, (0,1):1}, 'C': {(1,1):1}},
        {'A': {(0,1):1, (1,1):-1}, 'B': {(1,0):1, (1,1):1}, 'C': {(0,0):1}},
    ]
    
    outer_products = [
        {'A': [(0,1), (3,1)], 'B': [(0,1), (3,1)], 'C': [(0,1), (3,1)]},
        {'A': [(2,1), (3,1)], 'B': [(0,1)], 'C': [(2,1), (3,-1)]},
        {'A': [(0,1)], 'B': [(2,1), (3,-1)], 'C': [(1,1), (3,1)]},
        {'A': [(3,1)], 'B': [(1,1), (0,-1)], 'C': [(0,1), (2,1)]},
        {'A': [(0,1), (1,1)], 'B': [(3,1)], 'C': [(0,-1), (1,1)]},
        {'A': [(2,1), (0,-1)], 'B': [(0,1), (1,1)], 'C': [(3,1)]},
        {'A': [(1,1), (3,-1)], 'B': [(2,1), (3,1)], 'C': [(0,1)]},
    ]
    
    block_offsets = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
    
    # Process both halves: left (k=0..3) and right (k=4..7)
    for half in range(2):
        k_offset = half * 4  # 0 for left, 4 for right
        col_offset = half * 49  # 0 for left, 49 for right
        
        for op in outer_products:
            for ip in inner_products:
                col = col_offset  # This is wrong - need sequential counter
                # Let me use a proper counter
                pass
    
    # Redo with proper counter
    col = 0
    for half in range(2):
        k_offset = half * 4
        for op in outer_products:
            for ip in inner_products:
                # A entries
                for (ar, ac), a_ip in ip['A'].items():
                    for a_block, a_coeff in op['A']:
                        a_row = 2 * block_offsets[a_block][0] + ar
                        a_col = 2 * block_offsets[a_block][1] + ac
                        U[a_row * m + a_col][col] += a_coeff * a_ip
                
                # B entries
                for (br, bc), b_ip in ip['B'].items():
                    for b_block, b_coeff in op['B']:
                        b_row = 2 * block_offsets[b_block][0] + br
                        b_col = 2 * block_offsets[b_block][1] + bc + k_offset
                        V[b_row * p + b_col][col] += b_coeff * b_ip
                
                # C entries
                for c_block, c_coeff in op['C']:
                    for (cr, cc), c_ip in ip['C'].items():
                        c_row = 2 * block_offsets[c_block][0] + cr
                        c_col = 2 * block_offsets[c_block][1] + cc + k_offset
                        W[c_col * n + c_row][col] += c_coeff * c_ip
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
