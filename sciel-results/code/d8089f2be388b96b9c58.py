"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Strassen-based decomposition for 2x2 * 2x14 matrix multiplication.
    
    We split the 2x14 matrix into seven 2x2 blocks and apply Strassen's
    rank-7 algorithm to each block. This gives total rank 7*7 = 49.
    
    The trace convention is W[k*n+i, r] where n=2 (rows of A), p=14 (cols of B).
    A is n x m = 2 x 2, B is m x p = 2 x 14, C is n x p = 2 x 14.
    
    For each 2x2 block of B (columns 2*b to 2*b+1), we compute:
      C_block = A * B_block
    
    Strassen's 7 products for 2x2 * 2x2:
      M1 = (A11 + A22) * (B11 + B22)
      M2 = (A21 + A22) * B11
      M3 = A11 * (B12 - B22)
      M4 = A22 * (B21 - B11)
      M5 = (A11 + A12) * B22
      M6 = (A21 - A11) * (B11 + B12)
      M7 = (A12 - A22) * (B21 + B22)
      
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
    
    U has shape (n*m, rank) = (4, 49)
    V has shape (m*p, rank) = (28, 49)
    W has shape (p*n, rank) = (28, 49)
    
    Indexing:
      U[i*m + j, c]  where i in [0,2), j in [0,2)  -> row index in A
      V[j*p + k, c]  where j in [0,2), k in [0,14) -> row index in B
      W[k*n + i, c]  where k in [0,14), i in [0,2) -> output index (trace convention)
    
    For block b (0..6), B columns are k = 2*b and k = 2*b+1.
    Within a block, B11 is row 0 col 0 of the block -> j=0, k=2*b
                   B12 is row 0 col 1 of the block -> j=0, k=2*b+1
                   B21 is row 1 col 0 of the block -> j=1, k=2*b
                   B22 is row 1 col 1 of the block -> j=1, k=2*b+1
    """
    n, m, p = DIMENSIONS
    num_blocks = p // m  # 7 blocks
    rank = 7 * num_blocks  # 49
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49
    
    # Strassen coefficients for each of the 7 products:
    # Each product is (linear combo of A entries) * (linear combo of B entries)
    # A entries indexed by (i, j) where i = row (0,1), j = col (0,1)
    # B entries within a block indexed by (j, k_local) where j = row (0,1), k_local = col (0,1)
    
    # For product r (0..6):
    #   U part: coefficients for A[i*m + j]
    #   V part: coefficients for B[j*p + k]
    #   W part: coefficients for C output C[i_out][k_out] -> W[k_out * n + i_out]
    
    # Define the 7 Strassen products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # A indices: A11=(0,0), A12=(0,1), A21=(1,0), A22=(1,1)
    # A index in U: i*m + j
    # A11 -> 0*2+0 = 0
    # A12 -> 0*2+1 = 1
    # A21 -> 1*2+0 = 2
    # A22 -> 1*2+1 = 3
    
    # For each product, define:
    #   a_coeffs: dict from A_index to coefficient
    #   b_coeffs: dict from (b_row, b_col_local) to coefficient
    #   c_coeffs: dict from (c_row, c_col_local) to coefficient
    
    products = [
        # M1 = (A11 + A22) * (B11 + B22)
        {
            'a': {0: 1, 3: 1},       # A11 + A22
            'b': {(0,0): 1, (1,1): 1},  # B11 + B22
            'c': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}  # C11 += M1, C22 += M1
        },
        # M2 = (A21 + A22) * B11
        {
            'a': {2: 1, 3: 1},       # A21 + A22
            'b': {(0,0): 1},          # B11
            'c': {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}  # C21 += M2
        },
        # M3 = A11 * (B12 - B22)
        {
            'a': {0: 1},             # A11
            'b': {(0,1): 1, (1,1): -1},  # B12 - B22
            'c': {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}  # C12 += M3, C22 += M3
        },
        # M4 = A22 * (B21 - B11)
        {
            'a': {3: 1},             # A22
            'b': {(1,0): 1, (0,0): -1},  # B21 - B11
            'c': {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}  # C11 += M4, C21 += M4
        },
        # M5 = (A11 + A12) * B22
        {
            'a': {0: 1, 1: 1},       # A11 + A12
            'b': {(1,1): 1},          # B22
            'c': {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}  # C11 -= M5, C12 += M5
        },
        # M6 = (A21 - A11) * (B11 + B12)
        {
            'a': {2: 1, 0: -1},      # A21 - A11
            'b': {(0,0): 1, (0,1): 1},  # B11 + B12
            'c': {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}  # C22 += M6
        },
        # M7 = (A12 - A22) * (B21 + B22)
        {
            'a': {1: 1, 3: -1},      # A12 - A22
            'b': {(1,0): 1, (1,1): 1},  # B21 + B22
            'c': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}  # C11 += M7
        },
    ]
    
    for b in range(num_blocks):
        for r in range(7):
            col = b * 7 + r
            prod = products[r]
            
            # Fill U
            for a_idx, coeff in prod['a'].items():
                U[a_idx][col] = coeff
            
            # Fill V
            for (b_row, b_col_local), coeff in prod['b'].items():
                k = 2 * b + b_col_local  # global column in B
                v_idx = b_row * p + k
                V[v_idx][col] = coeff
            
            # Fill W
            for (c_row, c_col_local), coeff in prod['c'].items():
                k = 2 * b + c_col_local  # global column in C
                w_idx = k * n + c_row
                W[w_idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
