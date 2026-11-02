"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 36 decomposition
    # Split m=4 into m1=2, m2=2
    # Split p=5 into p1=2, p2=2, p3=1
    
    # U shape: (n*m, rank) = (8, 36)
    # V shape: (m*p, rank) = (20, 36)
    # W shape: (p*n, rank) = (10, 36)
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_2x2x2(i_offset, j_offset, k_offset):
        nonlocal col
        # i_offset: row offset in A (0 or 2)
        # j_offset: col offset in A (0 or 2) -> Wait, A is 2x4. i is row, j is col.
        # A indices: i in 0..1, j in 0..3.
        # B indices: j in 0..3, k in 0..4.
        # C indices: i in 0..1, k in 0..4.
        
        # Block A_sub: rows i_offset..i_offset+1, cols j_offset..j_offset+1
        # Block B_sub: rows j_offset..j_offset+1, cols k_offset..k_offset+1
        # Output C_sub: rows i_offset..i_offset+1, cols k_offset..k_offset+1
        
        # Strassen terms:
        # S1 = (A00 + A11) * (B00 + B11)
        # S2 = (A10 + A11) * B00
        # S3 = A00 * (B10 - B11)
        # S4 = A11 * (B01 - B00)
        # S5 = (A00 + A01) * B11
        # S6 = (A10 - A00) * (B00 + B01)
        # S7 = (A01 - A11) * (B10 + B11)
        
        # A indices:
        # A00: U[i_offset*4 + j_offset, col]
        # A01: U[i_offset*4 + j_offset+1, col]
        # A10: U[(i_offset+1)*4 + j_offset, col]
        # A11: U[(i_offset+1)*4 + j_offset+1, col]
        
        # B indices:
        # B00: V[j_offset*5 + k_offset, col]
        # B01: V[j_offset*5 + k_offset+1, col]
        # B10: V[(j_offset+1)*5 + k_offset, col]
        # B11: V[(j_offset+1)*5 + k_offset+1, col]
        
        # C indices:
        # C00: W[k_offset*2 + i_offset, col]
        # C01: W[(k_offset+1)*2 + i_offset, col]
        # C10: W[k_offset*2 + i_offset+1, col]
        # C11: W[(k_offset+1)*2 + i_offset+1, col]
        
        # S1
        U[i_offset*4 + j_offset][col] = 1
        U[(i_offset+1)*4 + j_offset+1][col] = 1
        V[j_offset*5 + k_offset][col] = 1
        V[(j_offset+1)*5 + k_offset+1][col] = 1
        W[k_offset*2 + i_offset][col] = 1
        W[(k_offset+1)*2 + i_offset+1][col] = 1
        col += 1
        
        # S2
        U[(i_offset+1)*4 + j_offset][col] = 1
        U[(i_offset+1)*4 + j_offset+1][col] = 1
        V[j_offset*5 + k_offset][col] = 1
        W[(k_offset+1)*2 + i_offset+1][col] = 1
        col += 1
        
        # S3
        U[i_offset*4 + j_offset][col] = 1
        V[(j_offset+1)*5 + k_offset][col] = 1
        V[(j_offset+1)*5 + k_offset+1][col] = -1
        W[(k_offset+1)*2 + i_offset][col] = 1
        W[(k_offset+1)*2 + i_offset+1][col] = -1
        col += 1
        
        # S4
        U[(i_offset+1)*4 + j_offset+1][col] = 1
        V[j_offset*5 + k_offset+1][col] = 1
        V[j_offset*5 + k_offset][col] = -1
        W[k_offset*2 + i_offset][col] = 1
        W[(k_offset+1)*2 + i_offset+1][col] = -1
        col += 1
        
        # S5
        U[i_offset*4 + j_offset][col] = 1
        U[i_offset*4 + j_offset+1][col] = 1
        V[(j_offset+1)*5 + k_offset+1][col] = 1
        W[k_offset*2 + i_offset+1][col] = 1
        col += 1
        
        # S6
        U[(i_offset+1)*4 + j_offset][col] = 1
        U[i_offset*4 + j_offset][col] = -1
        V[j_offset*5 + k_offset][col] = 1
        V[j_offset*5 + k_offset+1][col] = 1
        W[k_offset*2 + i_offset+1][col] = 1
        W[(k_offset+1)*2 + i_offset+1][col] = 1
        col += 1
        
        # S7
        U[i_offset*4 + j_offset+1][col] = 1
        U[(i_offset+1)*4 + j_offset+1][col] = -1
        V[(j_offset+1)*5 + k_offset][col] = 1
        V[(j_offset+1)*5 + k_offset+1][col] = 1
        W[k_offset*2 + i_offset][col] = 1
        W[(k_offset+1)*2 + i_offset][col] = -1
        col += 1

    # Helper to add a 2x2x1 block
    def add_2x2x1(i_offset, j_offset, k_offset):
        nonlocal col
        # A_sub: 2x2, B_sub: 2x1, C_sub: 2x1
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # Term 1: A00 * B0
        U[i_offset*4 + j_offset][col] = 1
        V[j_offset*5 + k_offset][col] = 1
        W[k_offset*2 + i_offset][col] = 1
        col += 1
        
        # Term 2: A01 * B1
        U[i_offset*4 + j_offset+1][col] = 1
        V[(j_offset+1)*5 + k_offset][col] = 1
        W[k_offset*2 + i_offset][col] = 1
        col += 1
        
        # Term 3: A10 * B0
        U[(i_offset+1)*4 + j_offset][col] = 1
        V[j_offset*5 + k_offset][col] = 1
        W[k_offset*2 + i_offset+1][col] = 1
        col += 1
        
        # Term 4: A11 * B1
        U[(i_offset+1)*4 + j_offset+1][col] = 1
        V[(j_offset+1)*5 + k_offset][col] = 1
        W[k_offset*2 + i_offset+1][col] = 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2x2(0, 0, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2x2(0, 0, 2)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    add_2x2x1(0, 0, 4)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2x2(0, 2, 0)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2x2(0, 2, 2)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    add_2x2x1(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
