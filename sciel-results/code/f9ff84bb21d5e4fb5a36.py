"""Strassen-based rank-28 decomposition for 2x4 by 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """Strassen-based rank-28 decomposition for 2x4 * 4x4 matrix multiplication.
    
    Partitions A into [A00, A01] and B into four 2x2 blocks.
    Uses Strassen's algorithm for each 2x2 product.
    Total rank = 4 * 7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen coefficients for 2x2 matrices X, Y -> Z = XY
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y10-Y00)
    # M4 = X11(Y01-Y11)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y10)
    # M7 = (X01-X11)(Y11+Y01)
    
    # Z00 = M1 - M4 + M7
    # Z01 = M5 + M4
    # Z10 = M2 + M3
    # Z11 = M1 - M2 + M6

    # Indices for A (U)
    # A00: U[0], U[1], U[4], U[5]
    # A01: U[2], U[3], U[6], U[7]
    
    # Indices for B (V)
    # B00: V[0], V[1], V[4], V[5]
    # B01: V[2], V[3], V[6], V[7]
    # B10: V[8], V[9], V[12], V[13]
    # B11: V[10], V[11], V[14], V[15]

    # Indices for C (W)
    # C00: W[0], W[2], W[1], W[3]  (C[0,0], C[0,1], C[1,0], C[1,1])
    # C01: W[4], W[6], W[5], W[7]  (C[0,2], C[0,3], C[1,2], C[1,3])

    # Define Strassen terms for a given A block and B block
    # A_block: list of 4 indices [A00, A01, A10, A11]
    # B_block: list of 4 indices [B00, B01, B10, B11]
    def strassen_terms(A_idx, B_idx):
        a00, a01, a10, a11 = A_idx
        b00, b01, b10, b11 = B_idx
        return [
            (a00, a11, b00, b11, 1, 1),  # M1: (a00+a11)(b00+b11)
            (a10, a11, b00, None, 1, 1), # M2: (a10+a11)b00
            (a00, None, b10, b00, 1, -1),# M3: a00(b10-b00)
            (a11, None, b01, b11, 1, -1),# M4: a11(b01-b11)
            (a00, a01, b11, None, 1, 1), # M5: (a00+a01)b11
            (a10, a00, b00, b10, 1, -1, 1, 1), # M6: (a10-a00)(b00+b10) -- Wait, format needs to be consistent
            (a01, a11, b11, b01, 1, -1, 1, 1)  # M7: (a01-a11)(b11+b01)
        ]
        # Let's just hardcode the coefficients for U, V, W for each of the 7 products.
        
    # Better approach: define the 7 products explicitly for each block combination.
    
    # Block 0: A00 * B00
    A00_idx = [0, 1, 4, 5]
    B00_idx = [0, 1, 4, 5]
    # Block 1: A01 * B10
    A01_idx = [2, 3, 6, 7]
    B10_idx = [8, 9, 12, 13]
    # Block 2: A00 * B01
    B01_idx = [2, 3, 6, 7]
    # Block 3: A01 * B11
    B11_idx = [10, 11, 14, 15]

    def add_strassen_block(A_idx, B_idx, W_idx, base_col):
        a00, a01, a10, a11 = A_idx
        b00, b01, b10, b11 = B_idx
        c00, c01, c10, c11 = W_idx
        
        # M1
        U[a00][base_col+0] += 1
        U[a11][base_col+0] += 1
        V[b00][base_col+0] += 1
        V[b11][base_col+0] += 1
        W[c00][base_col+0] += 1
        W[c11][base_col+0] += 1
        
        # M2
        U[a10][base_col+1] += 1
        U[a11][base_col+1] += 1
        V[b00][base_col+1] += 1
        W[c10][base_col+1] += 1
        W[c11][base_col+1] -= 1
        
        # M3
        U[a00][base_col+2] += 1
        V[b10][base_col+2] += 1
        V[b00][base_col+2] -= 1
        W[c10][base_col+2] += 1
        
        # M4
        U[a11][base_col+3] += 1
        V[b01][base_col+3] += 1
        V[b11][base_col+3] -= 1
        W[c00][base_col+3] -= 1
        W[c01][base_col+3] += 1
        
        # M5
        U[a00][base_col+4] += 1
        U[a01][base_col+4] += 1
        V[b11][base_col+4] += 1
        W[c01][base_col+4] += 1
        
        # M6
        U[a10][base_col+5] += 1
        U[a00][base_col+5] -= 1
        V[b00][base_col+5] += 1
        V[b10][base_col+5] += 1
        W[c11][base_col+5] += 1
        
        # M7
        U[a01][base_col+6] += 1
        U[a11][base_col+6] -= 1
        V[b11][base_col+6] += 1
        V[b01][base_col+6] += 1
        W[c00][base_col+6] += 1

    # C0 = A00*B00 + A01*B10
    # W indices for C0: c00=W[0], c01=W[2], c10=W[1], c11=W[3]
    W_C0 = [0, 2, 1, 3]
    add_strassen_block(A00_idx, B00_idx, W_C0, 0)
    add_strassen_block(A01_idx, B10_idx, W_C0, 7)

    # C1 = A00*B01 + A01*B11
    # W indices for C1: c00=W[4], c01=W[6], c10=W[5], c11=W[7]
    W_C1 = [4, 6, 5, 7]
    add_strassen_block(A00_idx, B01_idx, W_C1, 14)
    add_strassen_block(A01_idx, B11_idx, W_C1, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
