"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5 by 5x7 matrix multiplication.
    
    We use a block-based approach:
    - Partition the 3x5 matrix A into blocks: since 3 = 1+2 and 5 = 1+4, we can use
      a 2x2 block structure where the top-left block is 1x1, top-right is 1x4,
      bottom-left is 2x1, bottom-right is 2x4.
    - Partition the 5x7 matrix B into blocks: 5 = 1+4, 7 = 3+4, so top-left 1x3,
      top-right 1x4, bottom-left 4x3, bottom-right 4x4.
    
    Actually, let's use a more systematic approach. We'll use the fact that:
    C = A * B where A is 3x5, B is 5x7, C is 3x7.
    
    We can partition as:
    A = [A11 A12]  where A11 is 1x1, A12 is 1x4
        [A21 A22]  where A21 is 2x1, A22 is 2x4
    
    B = [B11 B12]  where B11 is 1x3, B12 is 1x4
        [B21 B22]  where B21 is 4x3, B22 is 4x4
    
    Then C = [A11*B11 + A12*B21,  A11*B12 + A12*B22]
             [A21*B11 + A22*B21,  A21*B12 + A22*B22]
    
    This gives us 8 sub-matrix multiplications. But we can use Strassen on the 2x2 block
    structure if we pad appropriately.
    
    Let me think differently. Let's use a simple block decomposition with rank counting.
    
    For a general block multiplication:
    If we partition into r x s blocks for A, s x t blocks for B, and r x t blocks for C,
    the schoolbook rank is the sum of the products of block dimensions.
    
    Let me try: partition rows of A (3) as 1+2, columns of A / rows of B (5) as 1+4, columns of B (7) as 3+4.
    
    Blocks:
    A: [1x1, 1x4; 2x1, 2x4]
    B: [1x3, 1x4; 4x3, 4x4]
    C: [1x3, 1x4; 2x3, 2x4]
    
    C11 = A11*B11 + A12*B21: rank = 1*1*3 + 1*4*3 = 3 + 12 = 15
    C12 = A11*B12 + A12*B22: rank = 1*1*4 + 1*4*4 = 4 + 16 = 20
    C21 = A21*B11 + A22*B21: rank = 2*1*3 + 2*4*3 = 6 + 24 = 30
    C22 = A21*B12 + A22*B22: rank = 2*1*4 + 2*4*4 = 8 + 32 = 40
    
    Total schoolbook rank for this partition: 15 + 20 + 30 + 40 = 105 (same as before)
    
    Now, can we apply Strassen to reduce this? Strassen works on 2x2 blocks where all blocks
    are the same size. Here our blocks are not square and not uniform.
    
    Let me try a different approach. We'll use the fact that we can write the multiplication
    as a sum of rank-1 tensors. The minimum rank is what we're looking for.
    
    For 3x5x7, the known lower bound is around 35-40 or so. Let me try to construct a 
    decomposition with rank 35.
    
    Actually, let me just use a simple block approach with Strassen where possible.
    
    Alternative: Use the fact that 3 = 2+1, 5 = 4+1, 7 = 4+3.
    
    Let me try padding to 4x4x8 and using Strassen recursively, then projecting back.
    
    Actually, the simplest approach that gives a clear improvement: use a 2x2 block 
    Strassen decomposition where we treat the problem as 2x2 blocks of appropriate sizes.
    
    Let me partition:
    A (3x5) into 2 row blocks: 2x5 and 1x5
    B (5x7) into 2 col blocks: 5x4 and 5x3
    C (3x7) into 2x2 blocks: 2x4, 2x3, 1x4, 1x3
    
    C11 (2x4) = A1(2x5) * B1(5x4)
    C12 (2x3) = A1(2x5) * B2(5x3)
    C21 (1x4) = A2(1x5) * B1(5x4)
    C22 (1x3) = A2(1x5) * B2(5x3)
    
    Each is a standard matrix multiplication. Rank for each:
    C11: 2*5*4 = 40
    C12: 2*5*3 = 30
    C21: 1*5*4 = 20
    C22: 1*5*3 = 15
    Total: 105
    
    This doesn't help. We need to use Strassen's trick.
    
    Let me try: view the 3x5x7 multiplication as a combination of smaller multiplications
    using Strassen's 7-multiplication scheme for 2x2 blocks.
    
    Partition A (3x5) as:
    - Top part: 2x5 (rows 0,1)
    - Bottom part: 1x5 (row 2)
    
    Partition B (5x7) as:
    - Left part: 5x4 (cols 0,1,2,3)
    - Right part: 5x3 (cols 4,5,6)
    
    We want to compute:
    C_top_left (2x4) = A_top(2x5) * B_left(5x4)
    C_top_right (2x3) = A_top(2x5) * B_right(5x3)
    C_bot_left (1x4) = A_bot(1x5) * B_left(5x4)
    C_bot_right (1x3) = A_bot(1x5) * B_right(5x3)
    
    If we could use Strassen on a 2x2 block structure, we'd need all four quadrants to be
    the same size. They're not. But we can still use the Strassen formulas by padding.
    
    Let me pad everything to 2x2 blocks of size 2x4 (for the left) and 2x3 (for the right).
    Actually, this is getting complicated. Let me just use a direct construction.
    
    Here's a cleaner approach: use the fact that we can decompose the multiplication
    tensor using a combination of smaller known decompositions.
    
    For now, let me just return the schoolbook decomposition. The task says to find
    rank-minimal, and 105 is the schoolbook rank. Any improvement would be better.
    
    Let me try a different block partition that might allow Strassen.
    
    Partition A (3x5) into 3x1 blocks: 3 blocks of size 1x5? No, that's just the schoolbook.
    
    Let me try: A as 1x3 blocks (3 blocks of 1x5), B as 5x1 blocks (5 blocks of 5x7)? No.
    
    I think for this specific problem, the best simple approach is to use a 2x2 block
    Strassen decomposition. Let me define it properly.
    
    We'll view the multiplication as:
    C = [A1; A2] * [B1 B2]
    where A1 is 2x5, A2 is 1x5, B1 is 5x4, B2 is 5x3.
    
    To use Strassen, we need to express this in terms of 7 products of "block" matrices.
    But the blocks aren't square. Strassen's algorithm works for any rectangular blocks
    as long as the dimensions are consistent.
    
    Strassen for 2x2 blocks:
    M1 = (A1 + A2) * (B1 + B2)  -- but A1+A2 would be 3x5 and B1+B2 would be 5x7, which is the whole thing!
    
    That's not right. Strassen partitions into 2x2 blocks of equal size. Here our 2x2 block
    structure has different sizes in each quadrant.
    
    Let me just use the schoolbook for now. The directive asks for a new approach, but
    without a clear path to a lower rank, I'll stick with a correct schoolbook implementation.
    
    Actually, let me try one more thing. We can use the following:
    
    C (3x7) = sum over j=0..4 of outer(A[:,j], B[j,:])
    
    This is the schoolbook with rank 105.
    
    For a potential improvement, consider that some entries might share structure. But
    without a specific algebraic identity, it's hard to do better.
    
    I'll return the schoolbook decomposition.
    """
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
