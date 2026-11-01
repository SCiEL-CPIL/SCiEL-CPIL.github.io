"""Program for matrix multiplication tensor decomposition 2x2 * 2x6."""
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-21 tensor decomposition for (2x2) * (2x6) matrix multiplication
    using Strassen's algorithm applied to three 2x2 blocks of the right-hand matrix.
    
    The output follows W[k*n + i, r] convention (column-major for output).
    
    For each block j (j=0,1,2), the right-hand matrix B is partitioned as:
    B_j = [[b_{0,3j}, b_{0,3j+1}],
           [b_{1,3j}, b_{1,3j+1}]]
    
    A = [[a_{00}, a_{01}],
         [a_{10}, a_{11}]]
    
    C_j = [[c_{0,3j}, c_{0,3j+1}],
           [c_{1,3j}, c_{1,3j+1}]]
    
    Strassen's 7 products for each block:
    P1 = a00 * (b00 + b11)
    P2 = (a10 + a11) * b00
    P3 = (a00 - a10) * (b10 + b11)
    P4 = a11 * (b01 - b00)
    P5 = (a00 + a11) * (b00 - b11)
    P6 = (a01 - a11) * (b10 + b11)
    P7 = (a10 + a00) * b01
    
    Reconstruction:
    c00 = P5 + P3 + P1 - P4 - P7
    c01 = P4 + P5
    c10 = P2 + P3
    c11 = P1 - P2 + P4 + P6
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=6
    
    rank = 21  # 3 blocks * 7 Strassen products
    
    # U has shape (n*m, rank) = (4, 21)
    # U[i*m + j, r] where i is row of A, j is col of A
    U = [[0] * rank for _ in range(n * m)]
    
    # V has shape (m*p, rank) = (12, 21)
    # V[j*p + k, r] where j is row of B, k is col of B
    V = [[0] * rank for _ in range(m * p)]
    
    # W has shape (p*n, rank) = (12, 21)
    # W[k*n + i, r] where k is col of C, i is row of C
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen product definitions
    # Each product is (A_factor, B_factor, output_coefficients)
    # A_factor: dict mapping (i, j) -> coefficient for A[i,j]
    # B_factor: dict mapping (j, k) -> coefficient for B[j,k]
    # For each block j_block, B columns are 3*j_block and 3*j_block+1
    
    strassen_products = [
        # P1: a00 * (b00 + b11)
        {'A': {(0, 0): 1}, 'B': {(0, 0): 1, (1, 1): 1},
         'C': {(0, 0): 1, (1, 1): -1}},
        # P2: (a10 + a11) * b00
        {'A': {(1, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1},
         'C': {(1, 0): 1, (1, 1): -1}},
        # P3: (a00 - a10) * (b10 + b11)
        {'A': {(0, 0): 1, (1, 0): -1}, 'B': {(1, 0): 1, (1, 1): 1},
         'C': {(0, 0): 1, (1, 0): 1}},
        # P4: a11 * (b01 - b00)
        {'A': {(1, 1): 1}, 'B': {(0, 1): 1, (0, 0): -1},
         'C': {(0, 0): -1, (0, 1): 1, (1, 1): 1}},
        # P5: (a00 + a11) * (b00 - b11)
        {'A': {(0, 0): 1, (1, 1): 1}, 'B': {(0, 0): 1, (1, 1): -1},
         'C': {(0, 0): 1, (0, 1): 1}},
        # P6: (a01 - a11) * (b10 + b11)
        {'A': {(0, 1): 1, (1, 1): -1}, 'B': {(1, 0): 1, (1, 1): 1},
         'C': {(1, 1): 1}},
        # P7: (a10 + a00) * b01
        {'A': {(1, 0): 1, (0, 0): 1}, 'B': {(0, 1): 1},
         'C': {(0, 0): -1}},
    ]
    
    for j_block in range(3):
        col_offset = j_block * 7
        for prod_idx, prod in enumerate(strassen_products):
            col = col_offset + prod_idx
            
            # Fill U entries
            for (ai, aj), coeff in prod['A'].items():
                U[ai * m + aj][col] = coeff
            
            # Fill V entries (B column offset is 3*j_block)
            for (bj, bk), coeff in prod['B'].items():
                V[bj * p + (3 * j_block + bk)][col] = coeff
            
            # Fill W entries (output C column is 3*j_block + ck_col, row is ci)
            # W[k*n + i, r] where k is col of C, i is row of C
            for (ci, ck), coeff in prod['C'].items():
                W[(3 * j_block + ck) * n + ci][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
