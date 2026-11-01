"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    # Rank 39 decomposition using 5 Strassen blocks (2x2) and 1 Schoolbook block (2x1)
    n, m, p = DIMENSIONS
    # U shape: (n*m, R) = (4, R)
    # V shape: (m*p, R) = (22, R)
    # W shape: (p*n, R) = (22, R)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a block
    # For M_{2,2,2} (Strassen) at B columns k0, k1
    def add_strassen_block(k0, k1, start_col):
        c = start_col
        # M1 = (A00+A11)(B00+B11)
        U[0][c] = 1 # A00
        U[3][c] = 1 # A11
        V[0*p + k0][c] = 1 # B0, k0
        V[1*p + k1][c] = 1 # B1, k1
        # C00 += M1
        W[k0*n + 0][c] = 1
        # C01 -= M1
        W[k1*n + 0][c] = -1
        # C10 -= M1
        W[k0*n + 1][c] = -1
        # C11 += M1
        W[k1*n + 1][c] = 1
        c += 1
        
        # M2 = (A10+A11)B00
        U[2][c] = 1 # A10
        U[3][c] = 1 # A11
        V[0*p + k0][c] = 1 # B0, k0
        # C10 += M2
        W[k0*n + 1][c] = 1
        # C11 += M2
        W[k1*n + 1][c] = 1
        c += 1
        
        # M3 = A00(B01-B11)
        U[0][c] = 1 # A00
        V[0*p + k1][c] = 1 # B0, k1
        V[1*p + k1][c] = -1 # B1, k1
        # C01 += M3
        W[k1*n + 0][c] = 1
        # C11 -= M3
        W[k1*n + 1][c] = -1
        c += 1
        
        # M4 = A11(B10-B00)
        U[3][c] = 1 # A11
        V[1*p + k0][c] = 1 # B1, k0
        V[0*p + k0][c] = -1 # B0, k0
        # C00 -= M4
        W[k0*n + 0][c] = -1
        # C10 += M4
        W[k0*n + 1][c] = 1
        c += 1
        
        # M5 = (A00+A01)B11
        U[0][c] = 1 # A00
        U[1][c] = 1 # A01
        V[1*p + k1][c] = 1 # B1, k1
        # C01 += M5
        W[k1*n + 0][c] = 1
        # C11 += M5
        W[k1*n + 1][c] = 1
        c += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][c] = 1 # A10
        U[0][c] = -1 # A00
        V[0*p + k0][c] = 1 # B0, k0
        V[0*p + k1][c] = 1 # B0, k1
        # C00 += M6
        W[k0*n + 0][c] = 1
        # C11 -= M6
        W[k1*n + 1][c] = -1
        c += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][c] = 1 # A01
        U[3][c] = -1 # A11
        V[1*p + k0][c] = 1 # B1, k0
        V[1*p + k1][c] = 1 # B1, k1
        # C00 -= M7
        W[k0*n + 0][c] = -1
        # C10 -= M7
        W[k0*n + 1][c] = -1
        c += 1
        
        return c

    # Helper to add a block
    # For M_{2,2,1} (Schoolbook) at B column k0
    def add_schoolbook_block(k0, start_col):
        c = start_col
        # C00 = A00*B00 + A01*B10
        # Term 1: A00*B00 -> C00
        U[0][c] = 1
        V[0*p + k0][c] = 1
        W[k0*n + 0][c] = 1
        c += 1
        # Term 2: A01*B10 -> C00
        U[1][c] = 1
        V[1*p + k0][c] = 1
        W[k0*n + 0][c] = 1
        c += 1
        # C10 = A10*B00 + A11*B10
        # Term 3: A10*B00 -> C10
        U[2][c] = 1
        V[0*p + k0][c] = 1
        W[k0*n + 1][c] = 1
        c += 1
        # Term 4: A11*B10 -> C10
        U[3][c] = 1
        V[1*p + k0][c] = 1
        W[k0*n + 1][c] = 1
        c += 1
        return c

    # 5 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9)
    col = add_strassen_block(0, 1, col)
    col = add_strassen_block(2, 3, col)
    col = add_strassen_block(4, 5, col)
    col = add_strassen_block(6, 7, col)
    col = add_strassen_block(8, 9, col)
    
    # 1 Schoolbook block for column 10
    col = add_schoolbook_block(10, col)
    
    # Verify col == 39
    assert col == 39, f"Expected 39, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
