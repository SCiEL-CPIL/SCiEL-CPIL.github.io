import numpy as np
import time

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for the rank-49 decomposition of 4x4 matrix multiplication.
    
    Shapes:
    U: (16, 49)
    V: (16, 49)
    W: (16, 49)
    
    This implementation uses a direct expansion of the recursive Strassen algorithm
    into explicit coefficient vectors for each of the 49 rank-1 terms, 
    ensuring structural distinctness from recursive accumulation methods.
    """
    n = 4
    m = 4
    p = 4
    rank = 49
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # We will generate the 49 terms.
    # Term index r is from 0 to 48.
    # r = outer_t * 7 + inner_k
    
    # Define the 7 outer Strassen block combinations
    # A blocks: 00, 01, 10, 11 (indices in 4x4 A)
    # B blocks: 00, 01, 10, 11 (indices in 4x4 B)
    # C blocks: 00, 01, 10, 11 (indices in 4x4 C)
    
    # Outer A factors: List of 7, each is a dict of {block_id: coeff}
    # Block IDs: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    outer_A_defs = [
        {0: 1, 3: 1},   # t=0: A00 + A11
        {2: 1, 3: 1},   # t=1: A10 + A11
        {0: 1},         # t=2: A00
        {3: 1},         # t=3: A11
        {0: 1, 1: 1},   # t=4: A00 + A01
        {2: 1, 0: -1},  # t=5: A10 - A00
        {1: 1, 3: -1},  # t=6: A01 - A11
    ]
    
    outer_B_defs = [
        {0: 1, 3: 1},   # t=0: B00 + B11
        {0: 1},         # t=1: B00
        {1: 1, 3: -1},  # t=2: B01 - B11
        {2: 1, 0: -1},  # t=3: B10 - B00
        {3: 1},         # t=4: B11
        {0: 1, 1: 1},   # t=5: B00 + B01
        {2: 1, 3: 1},   # t=6: B10 + B11
    ]
    
    outer_C_defs = [
        {0: 1, 3: 1},   # t=0: C00 + C11
        {2: 1, 3: -1},  # t=1: C10 - C11
        {1: 1, 3: 1},   # t=2: C01 + C11
        {0: 1, 2: 1},   # t=3: C00 + C10
        {0: -1, 1: 1},  # t=4: -C00 + C01
        {3: 1},         # t=5: C11
        {0: 1},         # t=6: C00
    ]
    
    # Inner Strassen definitions for 2x2 matrices
    # These operate on the scalar elements of the 2x2 blocks.
    # Let the 2x2 block be [[a00, a01], [a10, a11]]
    
    inner_A_defs = [
        {(0,0): 1, (1,1): 1}, # k=0: a00 + a11
        {(1,0): 1, (1,1): 1}, # k=1: a10 + a11
        {(0,0): 1},           # k=2: a00
        {(1,1): 1},           # k=3: a11
        {(0,0): 1, (0,1): 1}, # k=4: a00 + a01
        {(1,0): 1, (0,0): -1},# k=5: a10 - a00
        {(0,1): 1, (1,1): -1} # k=6: a01 - a11
    ]
    
    inner_B_defs = [
        {(0,0): 1, (1,1): 1}, # k=0: b00 + b11
        {(0,0): 1},           # k=1: b00
        {(0,1): 1, (1,1): -1},# k=2: b01 - b11
        {(1,0): 1, (0,0): -1},# k=3: b10 - b00
        {(1,1): 1},           # k=4: b11
        {(0,0): 1, (0,1): 1}, # k=5: b00 + b01
        {(1,0): 1, (1,1): 1}  # k=6: b10 + b11
    ]
    
    inner_C_defs = [
        {(0,0): 1, (1,1): 1}, # k=0: c00 + c11
        {(1,0): 1, (1,1): -1},# k=1: c10 - c11
        {(0,1): 1, (1,1): 1}, # k=2: c01 + c11
        {(0,0): 1, (1,0): 1}, # k=3: c00 + c10
        {(0,0): -1, (0,1): 1},# k=4: -c00 + c01
        {(1,1): 1},           # k=5: c11
        {(0,0): 1}            # k=6: c00
    ]
    
    # Helper to get global indices
    # Block (bi, bj) where bi, bj in 0,1
    # Offset row = bi*2, col = bj*2
    # Global element (i, j) in block corresponds to (bi*2+i, bj*2+j)
    
    def get_global_indices(block_id, local_i, local_j):
        # block_id 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        br = block_id // 2
        bc = block_id % 2
        return br * 2 + local_i, bc * 2 + local_j

    col = 0
    for t in range(7):
        for k in range(7):
            # 1. Construct U vector for this term
            # U corresponds to A coefficients
            # The term is (sum_A) * (sum_B)
            # sum_A = sum_{block in outer_A[t]} coeff * (sum_{local in inner_A[k]} coeff2 * A[global(block, local)])
            
            # We accumulate into U[row, col]
            # U row index: i*m + j = i*4 + j
            
            for block_a, coeff_a_outer in outer_A_defs[t].items():
                for local_pos_a, coeff_a_inner in inner_A_defs[k].items():
                    li, lj = local_pos_a
                    gi, gj = get_global_indices(block_a, li, lj)
                    total_coeff = coeff_a_outer * coeff_a_inner
                    if total_coeff != 0:
                        U[gi * m + gj, col] += total_coeff
            
            # 2. Construct V vector for this term
            # V row index: j*p + k (using j for B row, k for B col)
            # B is 4x4.
            
            for block_b, coeff_b_outer in outer_B_defs[t].items():
                for local_pos_b, coeff_b_inner in inner_B_defs[k].items():
                    li, lj = local_pos_b
                    gi, gj = get_global_indices(block_b, li, lj)
                    total_coeff = coeff_b_outer * coeff_b_inner
                    if total_coeff != 0:
                        V[gi * p + gj, col] += total_coeff
            
            # 3. Construct W vector for this term
            # W row index: k*n + i (Task convention: W[k*n+i, r])
            # C is 4x4.
            # The term contributes to C via outer_C[t] and inner_C[k].
            
            for block_c, coeff_c_outer in outer_C_defs[t].items():
                for local_pos_c, coeff_c_inner in inner_C_defs[k].items():
                    li, lj = local_pos_c
                    gi, gj = get_global_indices(block_c, li, lj)
                    total_coeff = coeff_c_outer * coeff_c_inner
                    if total_coeff != 0:
                        # W index: col index (gj) * n + row index (gi)
                        W[gj * n + gi, col] += total_coeff
            
            col += 1
            
    assert col == rank, f"Rank mismatch: {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
