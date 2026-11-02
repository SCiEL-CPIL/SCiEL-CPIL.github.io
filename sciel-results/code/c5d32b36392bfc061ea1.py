"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Use Strassen decomposition for 4x4 x 4x6
    # Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    # Split B (4x6) into 2x2 blocks: B = [[B11, B12, B13], [B21, B22, B23]]
    # 
    # Strassen 7 multiplications for 2x2 x 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M5
    #
    # We need C(4x6), so 2 row-blocks x 3 col-blocks of 2x2
    # For each pair (i,j) of 2x2 blocks, we need 7 multiplications
    # But we can reuse the A-side combinations
    #
    # Total rank: 7 * 3 = 21 (one set of 7 for each of 3 column-blocks of B)
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 16 x 21
    V = [[0] * rank for _ in range(m * p)]  # 24 x 21
    W = [[0] * rank for _ in range(p * n)]  # 24 x 21
    
    # A is 4x4, indexed as A[i,j] with i=0..3, j=0..3
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4]
    # A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    
    # B is 4x6, indexed as B[j,k] with j=0..3, k=0..5
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B13 = B[0:2,4:6]
    # B21 = B[2:4,0:2], B22 = B[2:4,2:4], B23 = B[2:4,4:6]
    
    # C is 4x6, indexed as C[i,k] with i=0..3, k=0..5
    # C11 = C[0:2,0:2], C12 = C[0:2,2:4], C13 = C[0:2,4:6]
    # C21 = C[2:4,0:2], C22 = C[2:4,2:4], C23 = C[2:4,4:6]
    
    # For each column-block t (0,1,2) of B, we do 7 Strassen multiplications
    # The A-side linear combinations are the same for all t
    
    # A-side combinations (shared across all column-blocks):
    # SA1 = A11 + A22
    # SA2 = A21 + A22
    # SA3 = A11
    # SA4 = A22
    # SA5 = A11 + A12
    # SA6 = A21 - A11
    # SA7 = A12 - A22
    
    # For column-block t, B-side combinations:
    # SB1_t = B1_t + B2_t
    # SB2_t = B1_t
    # SB3_t = B2_t - B1_t  (wait, let me re-derive)
    
    # Let me redefine more carefully for column-block t:
    # B_top_t = B[0:2, 2t:2t+2]  (2x2)
    # B_bot_t = B[2:4, 2t:2t+2]  (2x2)
    
    # M1_t = (A11+A22)(B_top_t + B_bot_t)
    # M2_t = (A21+A22)B_top_t
    # M3_t = A11(B_bot_t - B_top_t)
    # M4_t = A22(B_bot_t - B_top_t)  # wait, this isn't right
    
    # Let me use the standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For our case with column-block t:
    # B11 -> B_top_t, B12 -> 0 (since we're only doing one 2x2 product at a time)
    # Actually, I'm overcomplicating this.
    
    # Let me just use 7 independent 2x2x2 Strassen decompositions for each of the 3 column-blocks
    # For column-block t, we compute a 2x2 x 2x2 product using 7 multiplications
    
    # For t in {0, 1, 2}:
    # A-side (same for all t):
    #   A11, A12, A21, A22 are 2x2 submatrices of A
    # B-side:
    #   B_top_t = B[0:2, 2t:2t+2]
    #   B_bot_t = B[2:4, 2t:2t+2]
    
    # Strassen 7 mults for A(2x2) x B_t(2x2) where B_t = [[B_top_t], [B_bot_t]]:
    # Wait, A is 4x4, not 2x2. I need to be more careful.
    
    # Let me restart with a cleaner formulation.
    
    # A is 4x4. I'll treat it as having 2x2 blocks.
    # For the product C = A * B where B is 4x6:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Using Strassen on 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
    
    # C = [[C11, C12, C13], [C21, C22, C23]] where each is 2x2
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # I can use Strassen to compute each 2x2x2x2 product (which gives 2x2 result)
    # But that's 6 such products, each requiring 7 multiplications = 42 total
    
    # Alternatively, I can share computations. Let me just use 42 for now and see if I can do better.
    # Actually, let me try to be smarter.
    
    # For each column-block t, I need to compute:
    # [C1t; C2t] = [A11, A12] * [B1t; B2t]
    # This is a 2x4 times 4x2 matrix product, giving 2x2.
    # I can use Strassen-like decomposition for this.
    
    # Let me just go with the straightforward approach: 6 blocks * 7 mults = 42 rank
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 16 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # Strassen coefficients for 2x2 x 2x2 -> 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M5
    
    # For block (row-block r, col-block c) of the 2x2 block matrix:
    # A_r is the 2x2 block of A at row-block r
    # B_c is the 2x2 block of B at col-block c
    # But A_r is 2x4 (2 rows, 4 cols) and B_c is 4x2 (4 rows, 2 cols)
    # So I need to split A_r into two 2x2 blocks and B_c into two 2x2 blocks
    
    # For row-block r (0 or 1) and col-block c (0,1,2):
    # A_r = [A_r1, A_r2] where A_r1, A_r2 are 2x2
    # B_c = [B_c1; B_c2] where B_c1, B_c2 are 2x2
    # Product: A_r * B_c = A_r1 * B_c1 + A_r2 * B_c2
    
    # I'll use Strassen for the combined 2x4 x 4x2 product
    # Actually, let me just use 2 separate Strassen decompositions for each (r,c)
    
    # This is getting complex. Let me just use the schoolbook for now but with a different structure
    # to ensure correctness, then optimize.
    
    # For now, let me use rank = n*m*p = 96 with a clean implementation
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
