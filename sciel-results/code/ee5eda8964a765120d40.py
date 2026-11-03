"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-style block decomposition for 4x4 * 4x7
    # Partition into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] (each 2x2)
    # B = [[B11, B12], [B21, B22]] where B11,B21 are 2x4, B12,B22 are 2x3
    # Result C = A*B is 4x7: [[C11, C12], [C21, C22]]
    # C11 = A11*B11 + A12*B21 (2x2 * 2x4 -> 2x4)
    # C12 = A11*B12 + A12*B22 (2x2 * 2x3 -> 2x3)
    # C21 = A21*B11 + A22*B21 (2x2 * 2x4 -> 2x4)
    # C22 = A21*B12 + A22*B22 (2x2 * 2x3 -> 2x3)
    
    # Use Strassen for each 2x2 * 2x4 or 2x2 * 2x3 block multiplication
    # Strassen for 2x2 * 2xk: rank 7 instead of 8
    
    n, m, p = 4, 4, 7
    
    # Build rank-7 Strassen decomposition for 2x2 * 2xk
    # For matrices X (2x2), Y (2xk):
    # M1 = (X11 + X22) * (Y11 + Y22)  -- Y11 is 2x(k1), Y22 is 2x(k2) where k1+k2=k
    # M2 = (X21 + X22) * Y11
    # M3 = X11 * (Y22 - Y11)
    # M4 = X22 * (Y11 - Y22)
    # M5 = (X11 + X12) * Y22
    # M6 = (X21 - X11) * (Y11 + Y22)
    # M7 = (X12 - X22) * (Y22 - Y11)
    # Result: 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M1
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M5
    
    # Actually, let me use the standard Strassen for 2x2*2x2 and extend
    # For 2x2 * 2xk, we can treat Y as two columns of 2x1 or 2x2 blocks
    
    # Simpler: use the known rank-7 decomposition for 2x2*2x2
    # Then for 2x2*2x4, apply to each pair of columns
    
    # Let me just construct the full decomposition directly using Strassen structure
    
    # For 4x4 * 4x7, we have 4 block multiplications:
    # Each involves 2x2 * 2x4 or 2x2 * 2x3
    
    # Let me use a different approach: direct Strassen-style for the whole thing
    
    # Actually, the simplest is to use the schoolbook but with Strassen for 2x2 blocks
    
    # Let me construct rank-64 decomposition using Strassen for each 2x2*2x2 subproblem
    # But we have 2x4 and 2x3 blocks...
    
    # For 2x2 * 2x4: view as 2x2 * [2x2 | 2x2], so two Strassen decompositions
    # For 2x2 * 2x3: view as 2x2 * [2x2 | 2x1], one Strassen + one schoolbook
    
    # This is getting complex. Let me just use the known optimal for 4x4*4x7.
    
    # Known: rank(4x4*4x7) <= 64 via Strassen-style
    # Let me construct it properly
    
    # For multiplying (2a x 2b) * (2b x 2c) using Strassen: rank 7ab^2c? No...
    
    # Let me use the tensor product approach:
    # 4x4*4x7 = (2x2 tensor 2x2) * (2x2 tensor 2x2) with second factor being (2x2 tensor 2x1)
    
    # Actually, let me just return the schoolbook for now, as constructing Strassen
    # for non-square dimensions is error-prone. The score is rank, so we need to
    # actually reduce the rank.
    
    # Let me try: use Strassen for 2x2*2x2 (rank 7) and apply it to each 2x2 block
    # of the multiplication
    
    # A (4x4) = [[A11, A12], [A21, A22]]
    # B (4x7) = [[B11, B12, B13], [B21, B22, B23]] where Bij are 2x2 except B13,B23 are 2x1
    
    # C = A*B (4x7)
    # C11 = A11*B11 + A12*B21 (2x2)
    # C12 = A11*B12 + A12*B22 (2x2)
    # C13 = A11*B13 + A12*B23 (2x1)
    # C21 = A21*B11 + A22*B21 (2x2)
    # C22 = A21*B12 + A22*B22 (2x2)
    # C23 = A21*B13 + A22*B23 (2x1)
    
    # Use Strassen for each 2x2*2x2 (rank 7) and schoolbook for 2x2*2x1 (rank 4)
    # Total rank: 4*7 + 2*4 = 28 + 8 = 36
    
    # Wait, that's not right. Let me reconsider.
    
    # For C11 = A11*B11 + A12*B21: this is a sum of two 2x2 products
    # We can use Strassen on the combined computation
    
    # Actually, the correct approach: for each output block, we need to compute
    # X*Y + Z*W where all are 2x2. This can be done with rank 7*2 = 14? No...
    
    # Let me just use the schoolbook. It's rank 112.
    
    rank = n * m * p
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
