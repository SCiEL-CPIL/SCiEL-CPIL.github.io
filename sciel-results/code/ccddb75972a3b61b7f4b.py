"""Rewritten program for matrix multiplication 2,2,9 tensor decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # Rank 32: 4 Strassen blocks (2x2) + 1 Schoolbook block (2x1)
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]  # shape (4, rank)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (18, rank)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (18, rank)
    
    col = 0
    
    # Helper to add a Strassen block for B columns k0, k1
    def add_strassen_block(k0, k1):
        nonlocal col
        # A indices: 00, 01, 10, 11 -> U indices 0, 1, 2, 3
        # B indices: 
        # B_{0,k0} -> V[0*p+k0] = V[k0]
        # B_{0,k1} -> V[k1]
        # B_{1,k0} -> V[p+k0] = V[9+k0]
        # B_{1,k1} -> V[9+k1]
        
        # M1 = A00 * (B1k0 + B1k1)
        U[0][col] = 1.0
        V[p+k0][col] = 1.0
        V[p+k1][col] = 1.0
        # C00 += M1? No, C00 = M5 + M4 - M2 + M6. M1 is in C01.
        # C01 = M1 + M2
        W[k1*n + 0][col] = 1.0
        col += 1
        
        # M2 = (A00 + A01) * B1k1
        U[0][col] = 1.0
        U[1][col] = 1.0
        V[p+k1][col] = 1.0
        # C01 += M2
        W[k1*n + 0][col] = 1.0
        # C00 -= M2
        W[k0*n + 0][col] = -1.0
        col += 1
        
        # M3 = (A10 + A11) * B0k0
        U[2][col] = 1.0
        U[3][col] = 1.0
        V[k0][col] = 1.0
        # C10 += M3
        W[k0*n + 1][col] = 1.0
        # C11 += M3
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # M4 = A11 * (B0k0 - B1k0)
        U[3][col] = 1.0
        V[k0][col] = 1.0
        V[p+k0][col] = -1.0
        # C00 += M4
        W[k0*n + 0][col] = 1.0
        # C10 += M4
        W[k0*n + 1][col] = 1.0
        col += 1
        
        # M5 = (A00 + A11) * (B0k0 + B1k1)
        U[0][col] = 1.0
        U[3][col] = 1.0
        V[k0][col] = 1.0
        V[p+k1][col] = 1.0
        # C00 += M5
        W[k0*n + 0][col] = 1.0
        # C11 += M5
        W[k1*n + 1][col] = 1.0
        col += 1
        
        # M6 = (A01 - A11) * (B1k0 + B1k1)
        U[1][col] = 1.0
        U[3][col] = -1.0
        V[p+k0][col] = 1.0
        V[p+k1][col] = 1.0
        # C00 += M6
        W[k0*n + 0][col] = 1.0
        col += 1
        
        # M7 = (A00 - A10) * (B0k0 + B0k1)
        U[0][col] = 1.0
        U[2][col] = -1.0
        V[k0][col] = 1.0
        V[k1][col] = 1.0
        # C11 += M7
        W[k1*n + 1][col] = 1.0
        col += 1
        
    # Add 4 Strassen blocks for columns 0-7
    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    add_strassen_block(6, 7)
    
    # Add 1 Schoolbook block for column 8
    k0 = 8
    # C0 = A00*B0k0 + A01*B1k0
    # C1 = A10*B0k0 + A11*B1k0
    
    # P1 = A00 * B0k0
    U[0][col] = 1.0
    V[k0][col] = 1.0
    W[k0*n + 0][col] = 1.0
    col += 1
    
    # P2 = A01 * B1k0
    U[1][col] = 1.0
    V[p+k0][col] = 1.0
    W[k0*n + 0][col] = 1.0
    col += 1
    
    # P3 = A10 * B0k0
    U[2][col] = 1.0
    V[k0][col] = 1.0
    W[k0*n + 1][col] = 1.0
    col += 1
    
    # P4 = A11 * B1k0
    U[3][col] = 1.0
    V[p+k0][col] = 1.0
    W[k0*n + 1][col] = 1.0
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
