"""Program for matrix multiplication 2x2 by 2x11 tensor decomposition."""

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 39  # 5*7 + 1*4
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0] * rank for _ in range(p * n)]  # 22 x 39
    
    col = 0
    
    # Process 5 blocks of 2 columns using Strassen (rank 7 each)
    for block_idx in range(5):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        # Strassen's 7 multiplications for 2x2 matrix multiplication
        # Each multiplication M_i has specific U, V, W contributions
        
        # M1: (A00 + A11)(B00 + B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: contributes to C00 (+1), C11 (+1)
        U[0][col] += 1  # A00
        U[3][col] += 1  # A11
        V[0 * p + k0][col] += 1  # B00
        V[1 * p + k1][col] += 1  # B11
        W[k0 * n + 0][col] += 1  # C00
        W[k1 * n + 1][col] += 1  # C11
        col += 1
        
        # M2: (A10 + A11)(B00)
        # U: A10=1, A11=1
        # V: B00=1
        # W: contributes to C10 (+1), C11 (-1)
        U[2][col] += 1  # A10
        U[3][col] += 1  # A11
        V[0 * p + k0][col] += 1  # B00
        W[k0 * n + 1][col] += 1  # C10
        W[k1 * n + 1][col] += -1  # C11
        col += 1
        
        # M3: (A00)(B10 - B11)
        # U: A00=1
        # V: B10=1, B11=-1
        # W: contributes to C01 (+1), C11 (+1)
        U[0][col] += 1  # A00
        V[1 * p + k0][col] += 1  # B10
        V[1 * p + k1][col] += -1  # B11
        W[k1 * n + 0][col] += 1  # C01
        W[k1 * n + 1][col] += 1  # C11
        col += 1
        
        # M4: (A11)(B01 - B00)
        # U: A11=1
        # V: B01=1, B00=-1
        # W: contributes to C00 (+1), C10 (+1)
        U[3][col] += 1  # A11
        V[0 * p + k1][col] += 1  # B01
        V[0 * p + k0][col] += -1  # B00
        W[k0 * n + 0][col] += 1  # C00
        W[k0 * n + 1][col] += 1  # C10
        col += 1
        
        # M5: (A00 + A01)(B11)
        # U: A00=1, A01=1
        # V: B11=1
        # W: contributes to C01 (+1), C00 (-1)
        U[0][col] += 1  # A00
        U[1][col] += 1  # A01
        V[1 * p + k1][col] += 1  # B11
        W[k1 * n + 0][col] += 1  # C01
        W[k0 * n + 0][col] += -1  # C00
        col += 1
        
        # M6: (A10 - A00)(B00 + B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: contributes to C11 (+1)
        U[2][col] += 1  # A10
        U[0][col] += -1  # A00
        V[0 * p + k0][col] += 1  # B00
        V[0 * p + k1][col] += 1  # B01
        W[k1 * n + 1][col] += 1  # C11
        col += 1
        
        # M7: (A01 - A11)(B10 + B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: contributes to C00 (+1)
        U[1][col] += 1  # A01
        U[3][col] += -1  # A11
        V[1 * p + k0][col] += 1  # B10
        V[1 * p + k1][col] += 1  # B11
        W[k0 * n + 0][col] += 1  # C00
        col += 1
    
    # Process last block of 1 column using standard (rank 4)
    k0 = 10
    # N1: A00 * B00 -> C00
    U[0][col] += 1  # A00
    V[0 * p + k0][col] += 1  # B00
    W[k0 * n + 0][col] += 1  # C00
    col += 1
    
    # N2: A01 * B10 -> C01
    U[1][col] += 1  # A01
    V[1 * p + k0][col] += 1  # B10
    W[k0 * n + 0][col] += 1  # C01
    col += 1
    
    # N3: A10 * B00 -> C10
    U[2][col] += 1  # A10
    V[0 * p + k0][col] += 1  # B00
    W[k0 * n + 1][col] += 1  # C10
    col += 1
    
    # N4: A11 * B10 -> C11
    U[3][col] += 1  # A11
    V[1 * p + k0][col] += 1  # B10
    W[k0 * n + 1][col] += 1  # C11
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
