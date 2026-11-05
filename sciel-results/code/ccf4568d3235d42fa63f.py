"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # Use Strassen for 4x4x4 sub-problem (rank 49) + schoolbook for remaining 2 columns (rank 8)
    # Total rank: 57
    # 
    # For 4x4x4 Strassen:
    # U4: shape (16, 49)  -- 4x4 matrix A flattened
    # V4: shape (16, 49)  -- 4x4 matrix B flattened  
    # W4: shape (16, 49)  -- 4x4 matrix C flattened
    #
    # For 4x4x2 schoolbook:
    # U5: shape (16, 8)   -- 4x4 matrix A flattened
    # V5: shape (8, 8)    -- 4x2 matrix B flattened (cols 4,5)
    # W5: shape (8, 8)    -- 4x2 matrix C flattened (cols 4,5)
    #
    # Combined:
    # U: shape (12, 57)   -- remove last 4 rows (A row 3) from U4, keep U5 rows 0-11
    # V: shape (24, 57)   -- combine V4 (16 rows) and V5 (8 rows)
    # W: shape (18, 57)   -- remove last 8 rows (C cols 4,5 for row 3) from combined W
    
    # Strassen 4x4x4 coefficients (rank 49)
    # Standard Strassen for 4x4: divide into 2x2 blocks, apply Strassen recursively
    # This gives rank 49 = 7^3 for 2x2x2 applied to 4x4x4
    
    # Let me build the Strassen decomposition for 4x4x4 explicitly
    # The 4x4 matrix can be divided into 4 2x2 blocks
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #             [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Each 2x2*2x2 product uses Strassen with 7 multiplications
    # So total rank = 7 * 7 = 49 for the 4x4x4 problem
    
    # Define Strassen 2x2x2 decomposition (rank 7)
    # For 2x2 matrices:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U for 2x2x2 Strassen (shape 4x7, indices: A11=0, A12=1, A21=2, A22=3)
    U_2x2 = [
        [1, 0, 0, 0, 0, 0, 1],  # A11 in M1, M7
        [0, 1, 0, 0, 0, 1, 0],  # A12 in M3, M6
        [0, 0, 1, 0, 0, 1, 0],  # A21 in M2, M6
        [1, 0, 0, 1, 0, 0, 0],  # A22 in M1, M2
    ]
    
    # V for 2x2x2 Strassen (shape 4x7, indices: B11=0, B12=1, B21=2, B22=3)
    V_2x2 = [
        [1, 1, 0, 0, 0, 1, 0],  # B11 in M1, M2, M6
        [0, 1, 0, 0, 0, 1, 0],  # B12 in M3, M6
        [0, 0, 1, 0, 0, 0, 1],  # B21 in M5, M7
        [1, 0, 0, 1, 1, 0, 1],  # B22 in M1, M4, M5, M7
    ]
    
    # W for 2x2x2 Strassen (shape 4x7, indices: C11=0, C12=1, C21=2, C22=3)
    W_2x2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 1, 0, 0, 1, 0, 0],  # C12
        [0, 0, 1, 1, 0, 0, 0],  # C21
        [1, 0, 0, -1, 0, 1, 0], # C22
    ]
    
    # Now build 4x4x4 Strassen using 4x4x2x2 blocks
    # A is 4x4, indexed as A[i,j] for i,j in 0..3
    # Block structure: A11 = rows 0-1, cols 0-1; A12 = rows 0-1, cols 2-3; etc.
    
    # For 4x4x4, we have 4 sub-products each of rank 7, total rank 49
    # Sub-product 1: A11*B11 (uses A rows 0-1, cols 0-1 and B rows 0-1, cols 0-1, gives C rows 0-1, cols 0-1)
    # Sub-product 2: A12*B21 (uses A rows 0-1, cols 2-3 and B rows 2-3, cols 0-1, gives C rows 0-1, cols 2-3)
    # Sub-product 3: A21*B11 (uses A rows 2-3, cols 0-1 and B rows 0-1, cols 0-1, gives C rows 2-3, cols 0-1)
    # Sub-product 4: A22*B22 (uses A rows 2-3, cols 2-3 and B rows 2-3, cols 2-3, gives C rows 2-3, cols 2-3)
    
    # Wait, that's not quite right. Let me think more carefully.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each of these is a 2x2*2x2 product. So we need 4 Strassen decompositions.
    # But they share some structure. Actually, we can just concatenate them.
    
    # U4 shape: (16, 49)
    U4 = [[0] * 49 for _ in range(16)]
    V4 = [[0] * 49 for _ in range(16)]
    W4 = [[0] * 49 for _ in range(16)]
    
    # For each of the 4 sub-products, use 7 terms
    # Sub-product index s: 0=C11*A11*B11, 1=C11*A12*B21, 2=C12*A11*B12, 3=C12*A12*B22,
    #                      4=C21*A21*B11, 5=C21*A22*B21, 6=C22*A21*B12, 7=C22*A22*B22
    # Wait, that's 8 sub-products, but each C block is sum of 2 products.
    # C11 = A11*B11 + A12*B21 -> 2 products
    # C12 = A11*B12 + A12*B22 -> 2 products
    # C21 = A21*B11 + A22*B21 -> 2 products
    # C22 = A21*B12 + A22*B22 -> 2 products
    # Total: 8 products, each rank 7, total rank 56. Hmm, that's more than 49.
    
    # Actually, the standard Strassen for 4x4x4 has rank 49, not 56.
    # The key is that the 4x4x4 problem can be solved with rank 49 by a different decomposition.
    
    # Let me use a known result: the rank of 4x4x4 matrix multiplication is 49.
    # I'll construct it by embedding the 2x2x2 Strassen in a specific way.
    
    # Actually, let me just use the schoolbook for 4x4x4 (rank 64) and combine with schoolbook for 4x4x2 (rank 8), total rank 72. That's the same as the original.
    
    # Alternatively, use Strassen for 4x4x4 (rank 49) if I can construct it properly.
    
    # For now, let me try a simpler approach: use Strassen for the first 4x4x4 block (rank 49) and schoolbook for the remaining 2 columns (rank 8), total rank 57.
    
    # To construct 4x4x4 Strassen, I'll use the fact that it's equivalent to 7 2x2x2 Strassen products.
    
    # Let me define the 4x4x4 Strassen decomposition properly.
    # The 4x4 matrix A can be written as:
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13],
    #      [a20, a21, a22, a23],
    #      [a30, a31, a32, a33]]
    
    # The Strassen algorithm for 4x4x4 divides into 2x2 blocks and applies Strassen recursively.
    # This gives 7^3 = 343 operations for 8x8x8, and 7^2 = 49 for 4x4x4.
    
    # I'll construct the U, V, W matrices for 4x4x4 Strassen.
    
    # For simplicity, let me just use schoolbook for everything and get rank 72, which matches the seed.
    # Then I can try to improve.
    
    # Actually, let me just implement the schoolbook for 3x4x6 directly, which gives rank 72.
    
    rank = n * m * p  # 72
    U = [[0] * rank for _ in range(n * m)]  # (12, 72)
    V = [[0] * rank for _ in range(m * p)]  # (24, 72)
    W = [[0] * rank for _ in range(p * n)]  # (18, 72)
    
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
