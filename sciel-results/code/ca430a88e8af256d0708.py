import numpy as np

def construct():
    """
    Decompose 4x4 matrix multiplication using recursive Strassen.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    Rank is 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize U, V, W with zeros
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2 matrix multiplication
    # Each term is a list of (coeff, row, col) for the respective matrix factor
    # U factors (Left matrix A):
    strassen_U = [
        [(1, 0, 0), (1, 1, 1)],       # M1: A00 + A11
        [(1, 1, 0), (1, 1, 1)],       # M2: A10 + A11
        [(1, 0, 0)],                   # M3: A00
        [(1, 1, 1)],                   # M4: A11
        [(1, 0, 0), (1, 0, 1)],       # M5: A00 + A01
        [(1, 1, 0), (-1, 0, 0)],      # M6: A10 - A00
        [(1, 0, 1), (-1, 1, 1)],      # M7: A01 - A11
    ]
    
    # V factors (Right matrix B):
    strassen_V = [
        [(1, 0, 0), (1, 1, 1)],       # M1: B00 + B11
        [(1, 0, 0)],                   # M2: B00
        [(1, 0, 1), (-1, 1, 1)],      # M3: B01 - B11
        [(1, 1, 0), (-1, 0, 0)],      # M4: B10 - B00
        [(1, 1, 1)],                   # M5: B11
        [(1, 0, 0), (1, 0, 1)],       # M6: B00 + B01
        [(1, 1, 0), (1, 1, 1)],       # M7: B10 + B11
    ]
    
    # W factors (Result matrix C):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_W = [
        [(1, 0, 0), (1, 1, 1)],       # M1: C00 += 1, C11 += 1
        [(1, 1, 0), (-1, 1, 1)],      # M2: C10 += 1, C11 += -1
        [(1, 0, 1), (1, 1, 1)],       # M3: C01 += 1, C11 += 1
        [(1, 0, 0), (1, 1, 0)],       # M4: C00 += 1, C10 += 1
        [(-1, 0, 0), (1, 0, 1)],      # M5: C00 += -1, C01 += 1
        [(1, 1, 1)],                   # M6: C11 += 1
        [(1, 0, 0)],                   # M7: C00 += 1
    ]
    
    # The 4x4 multiplication is viewed as a 2x2 block multiplication
    # where each block is 2x2.
    # Outer Strassen operates on the 2x2 block structure.
    # Inner Strassen operates on the 2x2 scalar structure within each block.
    
    # Outer Strassen factors are identical in structure to inner ones
    # because the block structure is also 2x2.
    
    col_idx = 0
    for outer_s in range(7):
        for inner_s in range(7):
            # Combine outer and inner factors for U
            for (outer_coeff, outer_br, outer_bc) in strassen_U[outer_s]:
                for (inner_coeff, inner_lr, inner_lc) in strassen_U[inner_s]:
                    # Global row in A: 2 * outer_block_row + inner_local_row
                    # Global col in A: 2 * outer_block_col + inner_local_col
                    gr = 2 * outer_br + inner_lr
                    gc = 2 * outer_bc + inner_lc
                    U_idx = gr * m + gc
                    U[U_idx][col_idx] += outer_coeff * inner_coeff
            
            # Combine outer and inner factors for V
            for (outer_coeff, outer_br, outer_bc) in strassen_V[outer_s]:
                for (inner_coeff, inner_lr, inner_lc) in strassen_V[inner_s]:
                    # Global row in B: 2 * outer_block_row + inner_local_row
                    # Global col in B: 2 * outer_block_col + inner_local_col
                    gr = 2 * outer_br + inner_lr
                    gc = 2 * outer_bc + inner_lc
                    V_idx = gr * p + gc
                    V[V_idx][col_idx] += outer_coeff * inner_coeff
            
            # Combine outer and inner factors for W
            for (outer_coeff, outer_br, outer_bc) in strassen_W[outer_s]:
                for (inner_coeff, inner_lr, inner_lc) in strassen_W[inner_s]:
                    # Global row in C: 2 * outer_block_row + inner_local_row
                    # Global col in C: 2 * outer_block_col + inner_local_col
                    gr = 2 * outer_br + inner_lr
                    gc = 2 * outer_bc + inner_lc
                    # Convention: W[k*n + i] where k is col, i is row (based on seed analysis)
                    # Seed used: W[gc * n + gr]
                    W_idx = gc * n + gr
                    W[W_idx][col_idx] += outer_coeff * inner_coeff
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
