"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-56 exact decomposition for 2x4 * 4x8 matrix multiplication using block Strassen.
    
    A (2x4) is split into A1, A2 (2x2).
    B (4x8) is split into 4 column blocks of 4x2, each further split into B1j, B2j (2x2).
    C (2x8) is split into 4 column blocks Cj (2x2).
    Cj = A1 * B1j + A2 * B2j.
    
    We use Strassen's algorithm for each 2x2 * 2x2 product.
    Strassen uses 7 multiplications.
    There are 4 blocks j.
    For each j, we compute A1*B1j (7 mults) and A2*B2j (7 mults).
    Total = 4 * 14 = 56 multiplications.
    """
    n, m, p = DIMENSIONS
    # rank = 56
    rank = 56
    
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # Output C = [c00, c01, c10, c11]
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Indices for A (2x2 block):
    # A1: rows 0,1; cols 0,1
    # A2: rows 0,1; cols 2,3
    
    # Indices for B (2x2 block B1j):
    # B1j: rows 0,1; cols 2*j, 2*j+1
    # B2j: rows 2,3; cols 2*j, 2*j+1
    
    # We need to map the Strassen multiplications to the global U, V, W.
    # Each multiplication k in Strassen corresponds to a rank-1 term.
    # U vector is the linear combination of A entries.
    # V vector is the linear combination of B entries.
    # W vector places the result in C.
    
    # Let's define the 7 Strassen terms for a generic 2x2 A and 2x2 B.
    # A indices: 0,1,2,3 (row-major)
    # B indices: 0,1,2,3
    # C indices: 0,1,2,3
    
    # Term 1: (a0+a3)(b0+b3) -> contributes to c0, c3
    # Term 2: (a2+a3)b0 -> contributes to c2
    # Term 3: a0(b1-b3) -> contributes to c1
    # Term 4: a3(b2-b0) -> contributes to c0, c2
    # Term 5: (a0+a1)b3 -> contributes to c0, c1
    # Term 6: (a2-a0)(b0+b1) -> contributes to c3
    # Term 7: (a1-a3)(b2+b3) -> contributes to c0, c3
    
    # We will iterate over the 4 blocks j=0..3.
    # For each block, we have two products: A1*B1j and A2*B2j.
    # We will assign 14 columns of U,V,W per block.
    
    col = 0
    for j in range(4):
        # Process A1 * B1j
        # A1 is A[0:2, 0:2]
        # B1j is B[0:2, 2*j:2*j+2]
        
        # We need to fill U, V, W for the 7 Strassen terms.
        # For each term t (0..6):
        # U[global_row, col] = coeff_A
        # V[global_col, col] = coeff_B
        # W[global_out, col] = coeff_C
        
        # Helper to add a term
        def add_strassen_term(A_base, B_base, C_base, t):
            # A_base: index of A[0,0] in U
            # B_base: index of B[0,0] in V
            # C_base: index of C[0,0] in W
            nonlocal col
            # Term 0: (a0+a3)(b0+b3)
            if t == 0:
                U[A_base][col] = 1
                U[A_base+3][col] = 1
                V[B_base][col] = 1
                V[B_base+3][col] = 1
                W[C_base][col] = 1
                W[C_base+3][col] = 1
            # Term 1: (a2+a3)b0
            elif t == 1:
                U[A_base+2][col] = 1
                U[A_base+3][col] = 1
                V[B_base][col] = 1
                W[C_base+2][col] = 1
            # Term 2: a0(b1-b3)
            elif t == 2:
                U[A_base][col] = 1
                V[B_base+1][col] = 1
                V[B_base+3][col] = -1
                W[C_base+1][col] = 1
            # Term 3: a3(b2-b0)
            elif t == 3:
                U[A_base+3][col] = 1
                V[B_base+2][col] = 1
                V[B_base][col] = -1
                W[C_base][col] = 1
                W[C_base+2][col] = 1
            # Term 4: (a0+a1)b3
            elif t == 4:
                U[A_base][col] = 1
                U[A_base+1][col] = 1
                V[B_base+3][col] = 1
                W[C_base][col] = 1
                W[C_base+1][col] = 1
            # Term 5: (a2-a0)(b0+b1)
            elif t == 5:
                U[A_base+2][col] = 1
                U[A_base][col] = -1
                V[B_base][col] = 1
                V[B_base+1][col] = 1
                W[C_base+3][col] = 1
            # Term 6: (a1-a3)(b2+b3)
            elif t == 6:
                U[A_base+1][col] = 1
                U[A_base+3][col] = -1
                V[B_base+2][col] = 1
                V[B_base+3][col] = 1
                W[C_base][col] = 1
                W[C_base+3][col] = 1
            
            col += 1

        # A1 base index in U: 0 (A[0,0])
        # B1j base index in V: 0 + j*2 (B[0, 2j])
        # Cj base index in W: 0 + j*2 (C[0, 2j])
        
        for t in range(7):
            add_strassen_term(0, j*2, j*2, t)
            
        # Process A2 * B2j
        # A2 is A[0:2, 2:4] -> base index 2
        # B2j is B[2:4, 2*j:2*j+2] -> base index 2 + j*2
        # Cj base index in W: same as above
        
        for t in range(7):
            add_strassen_term(2, 2 + j*2, j*2, t)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
