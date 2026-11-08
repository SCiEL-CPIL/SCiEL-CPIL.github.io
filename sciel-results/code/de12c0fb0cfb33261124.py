"""Program for matrix multiplication tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will use a block decomposition with rank 46
    # Blocks:
    # C11 (2x2) from A11(2x2)B11(2x2) + A12(2x2)B21(2x2) -> Rank 14 (2x Strassen)
    # C12 (2x3) from A11(2x2)B12(2x3) + A12(2x2)B22(2x3) -> Rank 12 (2x Trivial 2x2x3)
    # C21 (1x2) from A21(1x2)B11(2x2) + A22(1x2)B21(2x2) -> Rank 8 (2x Trivial 1x2x2)
    # C22 (1x3) from A21(1x2)B12(2x3) + A22(1x2)B22(2x3) -> Rank 12 (2x Trivial 1x2x3)
    # Total Rank = 14 + 12 + 8 + 12 = 46
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a trivial decomposition for A_sub (na x ma) B_sub (ma x pa)
    # Indices in U: rows i, cols j
    # Indices in V: rows j, cols k
    # Indices in W: rows k, cols i
    def add_trivial(U, V, W, col, na, ma, pa, i_off, j_off, k_off):
        for i in range(na):
            for j in range(ma):
                for k in range(pa):
                    U[i_off + i][j_off + j][col] = 1
                    V[j_off + j][k_off + k][col] = 1
                    W[k_off + k][i_off + i][col] = 1
                    col += 1
        return col

    # Helper to add Strassen decomposition for 2x2x2
    # A is 2x2, B is 2x2
    # i_off, j_off, k_off are offsets
    def add_strassen(U, V, W, col, i_off, j_off, k_off):
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We define the linear forms for U, V, W for each of the 7 multiplications
        # U entries correspond to A coefficients
        # V entries correspond to B coefficients
        # W entries correspond to C coefficients
        
        # M1: U has 1 at (0,0) and (1,1) relative to block?
        # Let's map global indices.
        # A block rows: i_off, i_off+1
        # A block cols: j_off, j_off+1
        # B block rows: j_off, j_off+1
        # B block cols: k_off, k_off+1
        # C block rows: k_off, k_off+1
        # C block cols: i_off, i_off+1
        
        # M1
        U[i_off][j_off][col] = 1
        U[i_off+1][j_off+1][col] = 1
        V[j_off][k_off][col] = 1
        V[j_off+1][k_off+1][col] = 1
        # C11 += M1, C22 += M1
        W[k_off][i_off][col] = 1
        W[k_off+1][i_off+1][col] = 1
        col += 1
        
        # M2
        U[i_off+1][j_off][col] = 1
        U[i_off+1][j_off+1][col] = 1
        V[j_off][k_off][col] = 1
        # C21 += M2
        W[k_off][i_off+1][col] = 1
        col += 1
        
        # M3
        U[i_off][j_off][col] = 1
        V[j_off][k_off+1][col] = 1
        V[j_off+1][k_off+1][col] = -1
        # C12 += M3
        W[k_off+1][i_off][col] = 1
        col += 1
        
        # M4
        U[i_off+1][j_off+1][col] = 1
        V[j_off+1][k_off][col] = 1
        V[j_off][k_off][col] = -1
        # C11 += M4, C21 += M4
        W[k_off][i_off][col] = 1
        W[k_off][i_off+1][col] = 1
        col += 1
        
        # M5
        U[i_off][j_off][col] = 1
        U[i_off][j_off+1][col] = 1
        V[j_off+1][k_off+1][col] = 1
        # C11 -= M5, C12 += M5
        W[k_off][i_off][col] = -1
        W[k_off+1][i_off][col] = 1
        col += 1
        
        # M6
        U[i_off+1][j_off][col] = 1
        U[i_off][j_off][col] = -1
        V[j_off][k_off][col] = 1
        V[j_off][k_off+1][col] = 1
        # C22 += M6
        W[k_off+1][i_off+1][col] = 1
        col += 1
        
        # M7
        U[i_off][j_off+1][col] = 1
        U[i_off+1][j_off+1][col] = -1
        V[j_off+1][k_off][col] = 1
        V[j_off+1][k_off+1][col] = 1
        # C11 += M7, C22 += M7
        W[k_off][i_off][col] = 1
        W[k_off+1][i_off+1][col] = 1
        col += 1
        
        return col

    # Block 1: C11 (2x2) = A11(2x2)B11(2x2) + A12(2x2)B21(2x2)
    # This is sum of two 2x2x2 multiplications.
    # We can use Strassen for each.
    # A11 is at (0,0), A12 is at (0,2)
    # B11 is at (0,0), B21 is at (2,0)
    # C11 is at (0,0)
    
    # Strassen for A11 B11
    # i_off=0, j_off=0, k_off=0
    col = add_strassen(U, V, W, col, 0, 0, 0)
    
    # Strassen for A12 B21
    # A12 is rows 0-1, cols 2-3. i_off=0, j_off=2.
    # B21 is rows 2-3, cols 0-1. j_off=2, k_off=0.
    # C11 is rows 0-1, cols 0-1. k_off=0, i_off=0.
    # Note: The Strassen function assumes standard 2x2 layout.
    # Here A12 is 2x2, B21 is 2x2.
    col = add_strassen(U, V, W, col, 0, 2, 0)
    
    # Block 2: C12 (2x3) = A11(2x2)B12(2x3) + A12(2x2)B22(2x3)
    # A11: i_off=0, j_off=0. B12: j_off=0, k_off=2. C12: k_off=2, i_off=0.
    # A12: i_off=0, j_off=2. B22: j_off=2, k_off=2. C12: k_off=2, i_off=0.
    
    # Trivial for A11 B12 (2x2x3)
    col = add_trivial(U, V, W, col, 2, 2, 3, 0, 0, 2)
    
    # Trivial for A12 B22 (2x2x3)
    col = add_trivial(U, V, W, col, 2, 2, 3, 0, 2, 2)
    
    # Block 3: C21 (1x2) = A21(1x2)B11(2x2) + A22(1x2)B21(2x2)
    # A21: i_off=2, j_off=0. B11: j_off=0, k_off=0. C21: k_off=0, i_off=2.
    # A22: i_off=2, j_off=2. B21: j_off=2, k_off=0. C21: k_off=0, i_off=2.
    
    # Trivial for A21 B11 (1x2x2)
    col = add_trivial(U, V, W, col, 1, 2, 2, 2, 0, 0)
    
    # Trivial for A22 B21 (1x2x2)
    col = add_trivial(U, V, W, col, 1, 2, 2, 2, 2, 0)
    
    # Block 4: C22 (1x3) = A21(1x2)B12(2x3) + A22(1x2)B22(2x3)
    # A21: i_off=2, j_off=0. B12: j_off=0, k_off=2. C22: k_off=2, i_off=2.
    # A22: i_off=2, j_off=2. B22: j_off=2, k_off=2. C22: k_off=2, i_off=2.
    
    # Trivial for A21 B12 (1x2x3)
    col = add_trivial(U, V, W, col, 1, 2, 3, 2, 0, 2)
    
    # Trivial for A22 B22 (1x2x3)
    col = add_trivial(U, V, W, col, 1, 2, 3, 2, 2, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
