import numpy as np
import time

def construct():
    """
    Constructs a rank-28 tensor decomposition for 2x4 by 4x4 matrix multiplication
    using Strassen's algorithm on 2x2 blocks.
    
    Returns:
        dict: U, V, W matrices as lists of lists.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize U, V, W as lists of zeros
    # U: (n*m) x rank = 8 x 28
    # V: (m*p) x rank = 16 x 28
    # W: (p*n) x rank = 8 x 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen components for 2x2 * 2x2 -> 2x2
    # Each component is a tuple of three dicts:
    #   u: {A_local_idx: coeff}
    #   v: {B_local_idx: coeff}
    #   w: {C_local_idx: coeff}
    # Local indices for 2x2:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    strassen_components = [
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11) -> C01+=1, C11+=1
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Define the 4 block multiplications
    # A is 2x4. Split into A0 (cols 0,1) and A1 (cols 2,3).
    # B is 4x4. Split into 2x2 blocks:
    # B00 (rows 0,1 cols 0,1), B01 (rows 0,1 cols 2,3)
    # B10 (rows 2,3 cols 0,1), B11 (rows 2,3 cols 2,3)
    # C is 2x4. Split into C0 (cols 0,1) and C1 (cols 2,3).
    
    # Terms: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # A0 * B00 -> C0
    # A1 * B10 -> C0
    # A0 * B01 -> C1
    # A1 * B11 -> C1
    
    block_terms = [
        (0, 0, 0, 0), # A0, B00, C0
        (2, 2, 0, 0), # A1, B10, C0
        (0, 0, 2, 2), # A0, B01, C1
        (2, 2, 2, 2), # A1, B11, C1
    ]
    
    current_rank = 0
    for (a_co, b_ro, b_co, c_co) in block_terms:
        for (u_pat, v_pat, w_pat) in strassen_components:
            # Fill U
            for a_idx, val in u_pat.items():
                # a_idx is local 0..3
                # local row = a_idx // 2, local col = a_idx % 2
                a_row = a_idx // 2
                a_col = a_idx % 2
                global_a_idx = a_row * m + (a_co + a_col)
                U[global_a_idx][current_rank] += val
            
            # Fill V
            for b_idx, val in v_pat.items():
                b_row = b_idx // 2
                b_col = b_idx % 2
                global_b_idx = (b_ro + b_row) * p + (b_co + b_col)
                V[global_b_idx][current_rank] += val
            
            # Fill W
            # W index convention: W[k*n + i] where k is col of C, i is row of C
            for c_idx, val in w_pat.items():
                c_row = c_idx // 2
                c_col = c_idx % 2
                global_c_col = c_co + c_col
                global_c_idx = global_c_col * n + c_row
                W[global_c_idx][current_rank] += val
            
            current_rank += 1
    
    return {"U": U, "V": V, "W": W}
