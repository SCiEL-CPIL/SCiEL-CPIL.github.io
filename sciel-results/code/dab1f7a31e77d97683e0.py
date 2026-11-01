import numpy as np
import time

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    
    Strategy:
    We use a block decomposition based on Strassen's algorithm for 2x2 matrix multiplication.
    The columns of the second matrix (dimension p=15) are grouped into 7 pairs and 1 single column.
    Each pair of columns is handled by a Strassen block of rank 7.
    The single remaining column is handled by a schoolbook block of rank 4.
    Total rank = 7 * 7 + 4 = 53.
    
    This is lower than the schoolbook rank of 60.
    """
    n, m, p = DIMENSIONS
    
    # Rank calculation
    num_pairs = p // 2  # 7
    num_singles = p % 2 # 1
    rank = num_pairs * 7 + num_singles * 4 # 49 + 4 = 53
    
    # Dimensions of the factor matrices
    dim_U = n * m  # 4
    dim_V = m * p  # 30
    dim_W = p * n  # 30
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(dim_U)]
    V = [[0.0] * rank for _ in range(dim_V)]
    W = [[0.0] * rank for _ in range(dim_W)]
    
    # Indices for U (A matrix):
    # 0: A00, 1: A01, 2: A10, 3: A11
    
    # Helper to add a Strassen block for a pair of columns (k0, k1)
    def add_strassen_block(base_r, k0, k1):
        # k0, k1 are column indices in B and C
        # B indices: j*p + k
        # C indices: k*n + i
        
        r1 = base_r
        r2 = base_r + 1
        r3 = base_r + 2
        r4 = base_r + 3
        r5 = base_r + 4
        r6 = base_r + 5
        r7 = base_r + 6
        
        # P1 = (A00 + A11) * (B0k0 + B1k1)
        # Note: Standard Strassen usually uses (A00+A10)(B00+B11) etc, but we can permute.
        # Let's stick to the standard 7 products for 2x2:
        # m1 = (A00 + A10)(B00 + B11)
        # m2 = (A10 + A11)B00
        # m3 = A00(B01 - B11)
        # m4 = A11(B10 - B00)
        # m5 = (A00 + A01)B11
        # m6 = (A10 - A00)(B00 + B01)
        # m7 = (A01 - A11)(B10 + B11)
        
        # C00 = m1 + m4 - m5 + m7
        # C10 = m2 + m4
        # C01 = m3 + m5
        # C11 = m1 - m2 + m3 + m6
        
        # Here B00 corresponds to B[0, k0], B01 to B[0, k1], B10 to B[1, k0], B11 to B[1, k1]
        
        # m1
        U[0][r1] += 1.0 # A00
        U[2][r1] += 1.0 # A10
        V[0*p + k0][r1] += 1.0 # B0k0
        V[1*p + k1][r1] += 1.0 # B1k1
        
        # m2
        U[2][r2] += 1.0 # A10
        U[3][r2] += 1.0 # A11
        V[0*p + k0][r2] += 1.0 # B0k0
        
        # m3
        U[0][r3] += 1.0 # A00
        V[0*p + k1][r3] += 1.0 # B0k1
        V[1*p + k1][r3] += -1.0 # B1k1
        
        # m4
        U[3][r4] += 1.0 # A11
        V[1*p + k0][r4] += 1.0 # B1k0
        V[0*p + k0][r4] += -1.0 # B0k0
        
        # m5
        U[0][r5] += 1.0 # A00
        U[1][r5] += 1.0 # A01
        V[1*p + k1][r5] += 1.0 # B1k1
        
        # m6
        U[2][r6] += 1.0 # A10
        U[0][r6] += -1.0 # A00
        V[0*p + k0][r6] += 1.0 # B0k0
        V[0*p + k1][r6] += 1.0 # B0k1
        
        # m7
        U[1][r7] += 1.0 # A01
        U[3][r7] += -1.0 # A11
        V[1*p + k0][r7] += 1.0 # B1k0
        V[1*p + k1][r7] += 1.0 # B1k1
        
        # Output reconstruction
        # C[i, k] corresponds to W[k*n + i]
        
        # C0k0 (i=0, k=k0) -> W[k0*n + 0]
        W[k0*n + 0][r1] += 1.0
        W[k0*n + 0][r4] += 1.0
        W[k0*n + 0][r5] += -1.0
        W[k0*n + 0][r7] += 1.0
        
        # C1k0 (i=1, k=k0) -> W[k0*n + 1]
        W[k0*n + 1][r2] += 1.0
        W[k0*n + 1][r4] += 1.0
        
        # C0k1 (i=0, k=k1) -> W[k1*n + 0]
        W[k1*n + 0][r3] += 1.0
        W[k1*n + 0][r5] += 1.0
        
        # C1k1 (i=1, k=k1) -> W[k1*n + 1]
        W[k1*n + 1][r1] += 1.0
        W[k1*n + 1][r2] += -1.0
        W[k1*n + 1][r3] += 1.0
        W[k1*n + 1][r6] += 1.0

    # Add Strassen blocks for pairs
    col_offset = 0
    for i in range(num_pairs):
        k0 = 2 * i
        k1 = 2 * i + 1
        add_strassen_block(col_offset, k0, k1)
        col_offset += 7
        
    # Add schoolbook block for the single column
    if num_singles > 0:
        k_last = 2 * num_pairs
        r1 = col_offset
        r2 = col_offset + 1
        r3 = col_offset + 2
        r4 = col_offset + 3
        
        # Q1 = A00 B0k
        U[0][r1] += 1.0
        V[0*p + k_last][r1] += 1.0
        W[k_last*n + 0][r1] += 1.0
        
        # Q2 = A01 B1k
        U[1][r2] += 1.0
        V[1*p + k_last][r2] += 1.0
        W[k_last*n + 0][r2] += 1.0
        
        # Q3 = A10 B0k
        U[2][r3] += 1.0
        V[0*p + k_last][r3] += 1.0
        W[k_last*n + 1][r3] += 1.0
        
        # Q4 = A11 B1k
        U[3][r4] += 1.0
        V[1*p + k_last][r4] += 1.0
        W[k_last*n + 1][r4] += 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
