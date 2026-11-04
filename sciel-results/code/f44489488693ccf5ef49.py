"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll compute C = A * B where A is 4x4, B is 4x6, C is 4x6
    # Use a tensor decomposition approach:
    # Split the computation into parts that can be optimized
    # 
    # Approach: Use the fact that we can share some computations
    # For 4x4 * 4x6, we can think of it as computing 24 output entries
    # Each C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # Let's try a decomposition that exploits structure:
    # Split A into two 2x4 blocks: A_top (rows 0-1) and A_bottom (rows 2-3)
    # Then C_top = A_top * B (2x6) and C_bottom = A_bottom * B (2x6)
    #
    # For 2x4 * 4x6:
    # C[i][k] = sum_{j=0}^{3} A[i][j] * B[j][k]
    # This has 2*6 = 12 outputs, each requiring 4 multiplications = 48 total
    # But we can potentially share some.
    #
    # Actually, let me try a different approach: use the fact that matrix multiplication
    # tensors have known optimal decompositions for small sizes.
    #
    # For 2x2 * 2x2, Strassen gives rank 7.
    # For 2x2 * 2x4, we can think of it as two 2x2 * 2x2 products: rank 14
    # For 4x4 * 4x4, using 2-level Strassen: rank 49
    # For 4x4 * 4x6: 
    #   = 4x4 * 4x4 + 4x4 * 4x2
    #   = 49 + (using Strassen for 4x4 * 4x2)
    #
    # For 4x4 * 4x2:
    #   Split into 2x2 blocks
    #   A = [[A00, A01], [A10, A11]] (each 2x2)
    #   B = [[B00], [B10]] (each 2x2)
    #   C = [[C00], [C10]] (each 2x2)
    #
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    #
    # Using Strassen-like for each 2x2*2x2:
    # C00 needs 7 mults, C10 needs 7 mults, total 14
    # But we can share: B00 is used in both, B10 is used in both
    #
    # Actually, let me just implement a clean schoolbook and see if I can improve.
    # Schoolbook: rank = 4*4*6 = 96
    
    # Let me try to use a known better decomposition.
    # The rank of 4x4 * 4x4 is at most 49 (Strassen squared).
    # The rank of 4x4 * 4x2 is at most... let me compute.
    # 
    # For 4x4 * 4x2, we have 32 scalar multiplications in schoolbook.
    # Can we do better?
    # 
    # Let's use the block structure:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00], [B10]] where each is 2x2
    # 
    # C = [[A00*B00 + A01*B10],
    #      [A10*B00 + A11*B10]]
    #
    # Let M1 = A00*B00 (rank 7 with Strassen)
    # Let M2 = A01*B10 (rank 7 with Strassen)
    # Let M3 = A10*B00 (rank 7 with Strassen)
    # Let M4 = A11*B10 (rank 7 with Strassen)
    # Total: 28 for 4x4 * 4x2
    #
    # For 4x4 * 4x4: rank 49 (2-level Strassen)
    # For 4x4 * 4x2: rank 28
    # Total for 4x4 * 4x6: 49 + 28 = 77
    #
    # But wait, I need to be careful about how the blocks align.
    # B is 4x6, so B = [B_left | B_right] where B_left is 4x4 and B_right is 4x2
    # C = [C_left | C_right] where C_left = A*B_left (4x4) and C_right = A*B_right (4x2)
    #
    # So rank = 49 + 28 = 77
    #
    # Let me implement this.
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]  # 16 x 77
    V = [[0] * rank for _ in range(m * p)]  # 24 x 77
    W = [[0] * rank for _ in range(p * n)]  # 24 x 77
    
    col = 0
    
    # Helper: add a Strassen 2x2 x 2x2 decomposition
    # A is at rows iA..iA+1, cols jA..jA+1 of the full A (4x4)
    # B is at rows iB..iB+1, cols jB..jB+1 of the full B (4x6)
    # C is at rows iC..iC+1, cols jC..jC+1 of the full C (4x6)
    # W convention: W[k*n + i, r] where k is column of C, i is row of C
    def add_strassen_2x2(iA, jA, iB, jB, iC, jC, col, U, V, W):
        # M1 = (a00+a11)(b00+b11)
        U[iA * m + jA][col] = 1
        U[(iA + 1) * m + (jA + 1)][col] = 1
        V[iB * p + jB][col] = 1
        V[(iB + 1) * p + (jB + 1)][col] = 1
        W[jC * n + iC][col] = 1
        W[(jC + 1) * n + (iC + 1)][col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[(iA + 1) * m + jA][col] = 1
        U[(iA + 1) * m + (jA + 1)][col] = 1
        V[iB * p + jB][col] = 1
        W[jC * n + (iC + 1)][col] = 1
        W[(jC + 1) * n + (iC + 1)][col] = 1
        col += 1
        
        # M3 = a00(b01-b11)
        U[iA * m + jA][col] = 1
        V[iB * p + (jB + 1)][col] = 1
        V[(iB + 1) * p + (jB + 1)][col] = -1
        W[(jC + 1) * n + iC][col] = 1
        W[(jC + 1) * n + (iC + 1)][col] = -1
        col += 1
        
        # M4 = a11(b10-b00)
        U[(iA + 1) * m + (jA + 1)][col] = 1
        V[(iB + 1) * p + jB][col] = 1
        V[iB * p + jB][col] = -1
        W[jC * n + iC][col] = -1
        W[jC * n + (iC + 1)][col] = 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[iA * m + jA][col] = 1
        U[iA * m + (jA + 1)][col] = 1
        V[(iB + 1) * p + (jB + 1)][col] = 1
        W[jC * n + iC][col] = 1
        W[(jC + 1) * n + iC][col] = 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[(iA + 1) * m + jA][col] = 1
        U[iA * m + jA][col] = -1
        V[iB * p + jB][col] = 1
        V[iB * p + (jB + 1)][col] = 1
        W[(jC + 1) * n + (iC + 1)][col] = -1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[iA * m + (jA + 1)][col] = 1
        U[(iA + 1) * m + (jA + 1)][col] = -1
        V[(iB + 1) * p + jB][col] = 1
        V[(iB + 1) * p + (jB + 1)][col] = 1
        W[jC * n + iC][col] = -1
        col += 1
        
        return col
    
    # First part: A * B_left (4x4 * 4x4), result in C_left (cols 0-3)
    # Use 2-level Strassen: 7 block products, each 2x2*2x2 with rank 7
    # Total: 49
    
    # Block structure for 4x4 * 4x4:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M2 - M6
    
    # For each Mi, we need to compute a 2x2 * 2x2 product
    # But the "matrices" are sums/differences of blocks
    
    # Let me define a helper that adds a 2x2*2x2 product where
    # the A matrix has entries that are linear combinations of A's blocks
    # and the B matrix has entries that are linear combinations of B's blocks
    
    # Actually, let me just directly enumerate all 49 terms.
    # Each of the 7 block products is a 2x2*2x2, which uses 7 scalar products.
    # The A and B matrices for each block product are linear combinations.
    
    # This is complex. Let me use a simpler approach:
    # Just use schoolbook for the 4x4*4x4 part (rank 64) and 
    # schoolbook for the 4x4*4x2 part (rank 32). Total: 96.
    # But that's the same as the original.
    
    # Let me try a different split:
    # Use Strassen for the 4x4*4x4 part to get rank 49
    # Use schoolbook for the 4x4*4x2 part to get rank 32
    # Total: 81
    
    # Or even better:
    # For 4x4*4x2, use a smarter decomposition.
    
    # Let me implement the 49+32 = 81 approach first.
    
    # Actually, let me just go with schoolbook for now to ensure correctness,
    # then optimize.
    
    rank = n * m * p  # 96
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
