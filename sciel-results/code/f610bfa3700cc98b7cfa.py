"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Total rank: 3 blocks * (7 Strassen + 4 Outer) = 33
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1: A00+A11, B00+B11 -> C00, C11
    # M2: A10+A11, B00     -> C10, C11
    # M3: A00,       B01-B11-> C01, C11
    # M4: A11,       B10-B00-> C00, C10
    # M5: A00+A01,   B11    -> C00, C01
    # M6: A10-A00,   B00+B01-> C11
    # M7: A01-A11,   B10+B11-> C00, C11
    
    # Local indices for 2x2 block
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # We will iterate over the 3 blocks of B (and C)
    for b in range(3):
        # Local k indices in the block are 0, 1
        # Global k indices are 2*b, 2*b+1
        
        # --- Strassen Part (j=0,1) ---
        
        # M1
        U[0 * m + 0][col] = 1  # A00
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + 2*b][col] = 1  # B00 (global k=2b)
        V[1 * p + 2*b+1][col] = 1  # B11 (global k=2b+1)
        W[2*b * n + 0][col] = 1  # C00 (global k=2b, i=0)
        W[(2*b+1) * n + 1][col] = 1  # C11 (global k=2b+1, i=1)
        col += 1
        
        # M2
        U[1 * m + 0][col] = 1  # A10
        U[1 * m + 1][col] = 1  # A11
        V[0 * p + 2*b][col] = 1  # B00
        W[2*b * n + 1][col] = 1  # C10 (global k=2b, i=1)
        W[(2*b+1) * n + 1][col] = -1  # C11
        col += 1
        
        # M3
        U[0 * m + 0][col] = 1  # A00
        V[0 * p + 2*b+1][col] = 1  # B01
        V[1 * p + 2*b+1][col] = -1  # B11
        W[2*b * n + 0][col] = 1  # C01 (global k=2b+1, i=0)
        W[(2*b+1) * n + 1][col] = 1  # C11
        col += 1
        
        # M4
        U[1 * m + 1][col] = 1  # A11
        V[1 * p + 2*b][col] = 1  # B10
        V[0 * p + 2*b][col] = -1  # B00
        W[2*b * n + 0][col] = 1  # C00
        W[2*b * n + 1][col] = 1  # C10
        col += 1
        
        # M5
        U[0 * m + 0][col] = 1  # A00
        U[0 * m + 1][col] = 1  # A01
        V[1 * p + 2*b+1][col] = 1  # B11
        W[2*b * n + 0][col] = -1  # C00
        W[2*b * n + 0][col] = 1  # C01? No, C01 is k=2b+1, i=0.
        # Wait, C01 index is (2b+1)*n + 0.
        W[(2*b+1) * n + 0][col] = 1  # C01
        col += 1
        
        # M6
        U[1 * m + 0][col] = 1  # A10
        U[0 * m + 0][col] = -1  # A00
        V[0 * p + 2*b][col] = 1  # B00
        V[0 * p + 2*b+1][col] = 1  # B01
        W[(2*b+1) * n + 1][col] = 1  # C11
        col += 1
        
        # M7
        U[0 * m + 1][col] = 1  # A01
        U[1 * m + 1][col] = -1  # A11
        V[1 * p + 2*b][col] = 1  # B10
        V[1 * p + 2*b+1][col] = 1  # B11
        W[2*b * n + 0][col] = 1  # C00
        W[(2*b+1) * n + 1][col] = 1  # C11
        col += 1
        
        # --- Outer Product Part (j=2) ---
        # C_ik += A_i2 * B_2k
        # k in {2b, 2b+1}
        # i in {0, 1}
        
        # Term for i=0, k=2b
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + 2*b][col] = 1  # B_2, 2b
        W[2*b * n + 0][col] = 1  # C_2b, 0
        col += 1
        
        # Term for i=1, k=2b
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + 2*b][col] = 1  # B_2, 2b
        W[2*b * n + 1][col] = 1  # C_2b, 1
        col += 1
        
        # Term for i=0, k=2b+1
        U[0 * m + 2][col] = 1  # A02
        V[2 * p + 2*b+1][col] = 1  # B_2, 2b+1
        W[(2*b+1) * n + 0][col] = 1  # C_2b+1, 0
        col += 1
        
        # Term for i=1, k=2b+1
        U[1 * m + 2][col] = 1  # A12
        V[2 * p + 2*b+1][col] = 1  # B_2, 2b+1
        W[(2*b+1) * n + 1][col] = 1  # C_2b+1, 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
