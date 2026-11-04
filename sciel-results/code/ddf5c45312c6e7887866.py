"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # A_combos: 7 vectors of length 4 (indices 00, 01, 10, 11)
    A_combos = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    # B_combos: 7 vectors of length 4 (indices 00, 01, 10, 11)
    B_combos = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, -1, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # W_combos: 7 vectors of length 4 (indices C00, C01, C10, C11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    W_combos = [
        [1, 0, 0, 1], # M1
        [0, -1, 1, 0],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Define the 8 blocks
    # Block: (A_row_start, A_col_start, B_row_start, B_col_start)
    # A is 2x4. B is 4x8.
    # A_col_start: 0 or 2
    # B_row_start: 0 or 2
    # B_col_start: 0, 2, 4, 6
    
    blocks = []
    for a_col in [0, 2]:
        for b_row in [0, 2]:
            for b_col in [0, 2, 4, 6]:
                blocks.append((a_col, b_row, b_col))
                
    col_offset = 0
    for a_col, b_row, b_col in blocks:
        # Local indices
        # A local (i, j) -> Global (i, a_col + j)
        # B local (j, k) -> Global (b_row + j, b_col + k)
        # C local (i, k) -> Global (i, b_col + k)
        
        for m_idx in range(7):
            # A part
            for i in range(2):
                for j in range(2):
                    val = A_combos[m_idx][i * 2 + j]
                    if val != 0:
                        global_i = i
                        global_j = a_col + j
                        U[global_i * m + global_j][col_offset] = val
                        
            # B part
            for j in range(2):
                for k in range(2):
                    val = B_combos[m_idx][j * 2 + k]
                    if val != 0:
                        global_j = b_row + j
                        global_k = b_col + k
                        V[global_j * p + global_k][col_offset] = val
                        
            # W part
            for i in range(2):
                for k in range(2):
                    val = W_combos[m_idx][i * 2 + k]
                    if val != 0:
                        global_i = i
                        global_k = b_col + k
                        # W index: k * n + i
                        W[global_k * n + global_i][col_offset] = val
                        
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
