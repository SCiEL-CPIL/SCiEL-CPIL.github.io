import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation:
    # 2 blocks of (2x2x6) + (1x2x6)
    # 2x2x6 = 3 * (2x2x2) -> 3 * 7 = 21
    # 1x2x6 = 6 * (1x2x1) -> 6 * 2 = 12
    # Total per half-m = 33
    # Total rank = 2 * 33 = 66
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # Local indices 0,1 for i, j, k
        # Global indices:
        # U index: (i_off + i) * m + (j_off + j)
        # V index: (j_off + j) * p + (k_off + k)
        # W index: (k_off + k) * n + (i_off + i)
        
        # Term 1: M1 = A00 * (B11 - B01)
        # U: A00
        U[(i_off + 0) * m + (j_off + 0)][col] = 1
        # V: B11, B01
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        V[(j_off + 0) * p + (k_off + 1)][col] = -1
        # W: C00, C11
        W[(k_off + 0) * n + (i_off + 0)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # Term 2: M2 = (A00 + A01) * B00
        U[(i_off + 0) * m + (j_off + 0)][col] = 1
        U[(i_off + 0) * m + (j_off + 1)][col] = 1
        V[(j_off + 0) * p + (k_off + 0)][col] = 1
        W[(k_off + 1) * n + (i_off + 0)][col] = 1
        col += 1
        
        # Term 3: M3 = (A10 + A11) * B11
        U[(i_off + 1) * m + (j_off + 0)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[(k_off + 0) * n + (i_off + 1)][col] = 1
        col += 1
        
        # Term 4: M4 = A11 * (B01 - B00)
        U[(i_off + 1) * m + (j_off + 1)][col] = 1
        V[(j_off + 0) * p + (k_off + 1)][col] = 1
        V[(j_off + 0) * p + (k_off + 0)][col] = -1
        W[(k_off + 0) * n + (i_off + 0)][col] = 1
        col += 1
        
        # Term 5: M5 = (A00 + A10) * (B00 + B11)
        U[(i_off + 0) * m + (j_off + 0)][col] = 1
        U[(i_off + 1) * m + (j_off + 0)][col] = 1
        V[(j_off + 0) * p + (k_off + 0)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[(k_off + 0) * n + (i_off + 0)][col] = 1
        W[(k_off + 1) * n + (i_off + 0)][col] = 1
        W[(k_off + 0) * n + (i_off + 1)][col] = 1
        col += 1
        
        # Term 6: M6 = (A01 - A11) * (B10 + B11)
        U[(i_off + 0) * m + (j_off + 1)][col] = 1
        U[(i_off + 1) * m + (j_off + 1)][col] = -1
        V[(j_off + 1) * p + (k_off + 0)][col] = 1
        V[(j_off + 1) * p + (k_off + 1)][col] = 1
        W[(k_off + 1) * n + (i_off + 1)][col] = 1
        col += 1
        
        # Term 7: M7 = (A10 - A00) * (B00 + B01)
        U[(i_off + 1) * m + (j_off + 0)][col] = 1
        U[(i_off + 0) * m + (j_off + 0)][col] = -1
        V[(j_off + 0) * p + (k_off + 0)][col] = 1
        V[(j_off + 0) * p + (k_off + 1)][col] = 1
        W[(k_off + 0) * n + (i_off + 0)][col] = -1
        W[(k_off + 0) * n + (i_off + 1)][col] = 1
        col += 1

    def add_linear_block(i_idx, j_off, k_idx):
        nonlocal col
        # 1x2x1 block: C = A0*B0 + A1*B1
        # Term 1: A0 * B0 -> C
        U[i_idx * m + (j_off + 0)][col] = 1
        V[(j_off + 0) * p + k_idx][col] = 1
        W[k_idx * n + i_idx][col] = 1
        col += 1
        
        # Term 2: A1 * B1 -> C
        U[i_idx * m + (j_off + 1)][col] = 1
        V[(j_off + 1) * p + k_idx][col] = 1
        W[k_idx * n + i_idx][col] = 1
        col += 1

    # Loop over the two m-blocks (j offset 0 and 2)
    for j_off in [0, 2]:
        # 1. 2x2x6 part (i=0,1)
        # Split k into 3 blocks of 2
        for k_off in [0, 2, 4]:
            add_strassen_block(0, j_off, k_off)
            
        # 2. 1x2x6 part (i=2)
        # Loop over each k
        for k_idx in range(p):
            add_linear_block(2, j_off, k_idx)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
