<<<<<<< SEARCH
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
=======
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Hybrid decomposition: Strassen for 4x4*4x4 (rank 7) + Schoolbook for 4x4*4x3 (rank 12).
    Total rank = 7 + 12 = 19... wait, that's not right.
    
    Actually: For 4x4 * 4x7, we can split the 7 columns into 4+3.
    - First 4 columns: use Strassen 4x4*4x4 (rank 7 per 2x2 block, but we need 4x4)
    
    Let me reconsider. The known result is:
    - 4x4 * 4x4 can be done with Strassen-based decomposition
    - 4x4 * 4x3 can be done with schoolbook (rank 48)
    
    But a better approach: use the 2x2 Strassen algorithm applied to 2x2 blocks.
    A 4x4 matrix can be seen as 2x2 blocks of 2x2 matrices.
    Strassen on 2x2 blocks: rank 7 for each 2x2*2x2 multiplication.
    For 4x4 * 4x4: we have 4 block multiplications, each 2x2*2x2, so 4*7 = 28.
    
    For 4x4 * 4x7: split into 4x4*4x4 + 4x4*4x3.
    - 4x4*4x4: rank 28 (Strassen on 2x2 blocks)
    - 4x4*4x3: rank 48 (schoolbook)
    Total: 76
    
    Actually, let me use a more careful approach. The known minimal rank for 4x4*4x7 
    is at most 97 using a specific hybrid. Let me implement the straightforward approach:
    
    Use Strassen for the 4x4*4x4 part (rank 28) and schoolbook for 4x4*4x3 (rank 48).
    Total rank = 76.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Split p into p1=4 and p2=3
    p1 = 4
    p2 = p - p1  # 3
    
    # Part 1: Strassen decomposition for 4x4 * 4x4 (using 2x2 blocks)
    # 2x2 Strassen: 7 products
    # For 4x4*4x4 as 2x2 blocks of 2x2 matrices:
    # We need to map the 2x2 Strassen to 4x4 indices
    
    # 2x2 Strassen formulas for blocks A11,A12,A21,A22 and B11,B12,B21,B22:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4*4x4, we have 4 block products:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each block product is 2x2*2x2, which needs rank 7 with Strassen.
    # So total rank for 4x4*4x4 = 4 * 7 = 28
    
    # Part 2: Schoolbook for 4x4 * 4x3
    # Rank = 4 * 4 * 3 = 48
    
    total_rank = 28 + 48
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen part: 4x4 * 4x4 (columns 0-3 of the result)
    # 2x2 Strassen coefficients: each term has entries in {-1, 0, 1}
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each of the 4 block positions, we apply Strassen
    # Block (bi, bj) where bi, bj in {0,1}
    # A block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block (bj, bk) corresponds to rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    
    # Strassen terms for a given block multiplication:
    # Each term is a sum/difference of A entries times sum/difference of B entries
    
    # Let's define the 7 Strassen multipliers for 2x2:
    # For A block with sub-blocks A00,A01,A10,A11 and B block with B00,B01,B10,B11:
    # M1: (A00+A11) * (B00+B11)
    # M2: (A10+A11) * B00
    # M3: A00 * (B01-B11)
    # M4: A11 * (B10-B00)
    # M5: (A00+A01) * B11
    # M6: (A10-A00) * (B00+B01)
    # M7: (A01-A11) * (B10+B11)
    
    # For 4x4, we have 4 independent 2x2 block multiplications:
    # (0,0): A[0:2,0:2] * B[0:2,0:4] -> C[0:2,0:4]
    # (0,1): A[0:2,2:4] * B[2:4,0:4] -> C[0:2,0:4]  (adds to C[0:2,0:4])
    # Wait, this is getting complicated. Let me use a cleaner approach.
    
    # Actually, for 4x4*4x4, the standard approach is:
    # View as 2x2 blocks of 2x2 matrices.
    # C = A*B where A,B are 2x2 blocks of 2x2 matrices.
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each product like A11*B11 is a 2x2*2x2 multiplication.
    # We can apply Strassen to each, but they're independent.
    # So we have 8 independent 2x2*2x2 multiplications, each of rank 7.
    # Total rank = 8 * 7 = 56. That's worse than schoolbook (64)!
    
    # Hmm, that's not helpful. Let me reconsider.
    
    # Actually, the known result for 4x4*4x4 is rank 49 (using a different method)
    # or we can use the fact that 4x4*4x4 has rank at most 49.
    
    # Let me just use a simpler approach: schoolbook for the whole thing but with
    # some optimization. Actually, let me use the known hybrid:
    # 4x4 * 4x7 = 4x4*4x4 + 4x4*4x3
    # Use Strassen-like for 4x4*4x4 and schoolbook for 4x4*4x3.
    
    # For now, let me implement a clean schoolbook and verify it works, then optimize.
    
    # Reset: use pure schoolbook first to ensure correctness
    U = [[0] * (n*m*p) for _ in range(n * m)]
    V = [[0] * (n*m*p) for _ in range(m * p)]
    W = [[0] * (n*m*p) for _ in range(p * n)]
    
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
>>>>>>> REPLACE

I need to implement the actual Strassen-based hybrid. Let me think more carefully about the 4x4*4x4 Strassen decomposition.

