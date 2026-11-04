"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Matrix multiplication 4x5 * 5x5.
    
    Strategy: Use Strassen's algorithm on 2x2 sub-blocks where possible.
    For 4x5 * 5x5:
    - Split A (4x5) into 2x2 blocks of size 2x2 and 2x3 (top/bottom rows, left/right cols)
    - Split B (5x5) into 2x2 blocks of size 2x2 and 3x2, 2x3, 3x3
    - Split C (4x5) into 2x2 blocks of size 2x2 and 2x3
    
    Actually, let me think more carefully about the block structure.
    
    A is 4x5, B is 5x5, C = A*B is 4x5.
    
    Let's split:
    - A: top 2 rows, bottom 2 rows; left 2 cols, right 3 cols
      A = [[A11 (2x2), A12 (2x3)],
           [A21 (2x2), A22 (2x3)]]
    
    - B: top 2 rows, bottom 3 rows; left 2 cols, right 3 cols
      B = [[B11 (2x2), B12 (2x3)],
           [B21 (3x2), B22 (3x3)]]
    
    - C: top 2 rows, bottom 2 rows; left 2 cols, right 3 cols
      C = [[C11 (2x2), C12 (2x3)],
           [C21 (2x2), C22 (2x3)]]
    
    C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x3*3x2)
    C12 = A11*B12 + A12*B22  (2x3 = 2x2*2x3 + 2x3*3x3)
    C21 = A21*B11 + A22*B21  (2x2 = 2x2*2x2 + 2x3*3x2)
    C22 = A21*B12 + A22*B22  (2x3 = 2x2*2x3 + 2x3*3x3)
    
    This is getting complex. Let me use a different approach.
    
    The known best rank for 4x5*5x5 is likely achievable through block Strassen.
    
    Let me use a simpler approach: split into 2x2 and 3x3 blocks.
    
    Actually, let me try a different decomposition. Split A into 2+2 rows, B into 2+3 rows.
    
    A (4x5) = [A_top (2x5); A_bot (2x5)]
    B (5x5) = [B_top (2x5); B_bot (3x5)]
    C (4x5) = [C_top (2x5); C_bot (2x5)]
    
    C_top = A_top * B  (2x5 * 5x5 = 2x5)
    C_bot = A_bot * B  (2x5 * 5x5 = 2x5)
    
    This doesn't help directly. Let me think about using Strassen on 2x2 blocks.
    
    For the 2x2 * 2x2 part, we can use Strassen (rank 7 instead of 8).
    
    Let me identify all the 2x2*2x2 subproblems:
    C11 (2x2) = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    C21 (2x2) = A21(2x2)*B11(2x2) + A22(2x3)*B21(3x2)
    
    The 2x3*3x2 terms are rank 9 each (schoolbook), and 2x2*2x2 is rank 4.
    
    Hmm, this is getting complicated. Let me just implement the schoolbook and see if I can find a better decomposition.
    
    Actually, from the knowledge: "Using Strassen's algorithm for 2x2x2 sub-blocks reduces the rank from 100 to 92"
    
    So the approach is:
    1. Identify 2x2*2x2 sub-blocks in the multiplication
    2. Replace each with Strassen (rank 7 instead of 8)
    3. Keep other multiplications as schoolbook
    
    For 4x5 * 5x5:
    If we split A into 2x2, 2x3, 2x2, 2x3 blocks and B into 2x2, 2x3, 3x2, 3x3 blocks:
    
    The 2x2*2x2 products are:
    - A11(2x2) * B11(2x2) -> contributes to C11(2x2)
    - A21(2x2) * B11(2x2) -> contributes to C21(2x2)
    
    That's only 2 such products, saving 2*(8-7) = 2 rank. Not enough.
    
    Let me try a different split. What if I split differently to get more 2x2*2x2 blocks?
    
    Actually, let me just implement the known result: rank 92 via Strassen on 2x2 blocks.
    
    The key insight from the knowledge is that replacing 8 schoolbook 2x2 blocks with Strassen saves 8 ranks (100 -> 92).
    
    For this to work, we need 8 instances of 2x2*2x2 multiplication.
    
    Let me think about how to get 8 such instances. If we split A (4x5) and B (5x5) appropriately...
    
    A (4x5): split into 2x2, 2x2, 2x1 columns? No, that's awkward.
    
    Let me try: A (4x5) split as [A1 (2x2), A2 (2x2), A3 (2x1)]
    B (5x5) split as [B1 (2x5); B2 (2x5); B3 (1x5)]
    
    Then C (4x5) = A1*B1 + A2*B2 + A3*B3
    
    A1*B1: 2x2 * 2x5 = 2x5 (rank 10 schoolbook)
    A2*B2: 2x2 * 2x5 = 2x5 (rank 10 schoolbook)
    A3*B3: 2x1 * 1x5 = 2x5 (rank 5 schoolbook)
    
    Total: 25. Not helpful.
    
    Let me try yet another approach. The standard way to apply Strassen is to recursively split into equal-sized blocks. For 4x5*5x5, we can't split equally, but we can pad or use asymmetric splits.
    
    Actually, let me just implement the direct Strassen-based decomposition for 2x2 blocks within the larger problem.
    
    Here's my plan:
    1. Split A (4x5) into 4 blocks: A11(2x2), A12(2x3), A21(2x2), A22(2x3)
    2. Split B (5x5) into 4 blocks: B11(2x2), B12(2x3), B21(3x2), B22(3x3)
    3. Compute C blocks using Strassen where possible
    
    C11(2x2) = A11*B11 + A12*B21
    C12(2x3) = A11*B12 + A12*B22
    C21(2x2) = A21*B11 + A22*B21
    C22(2x3) = A21*B12 + A22*B22
    
    For C11 and C21, we have 2x2*2x2 + 2x3*3x2 terms.
    The 2x2*2x2 part can use Strassen (rank 7), and 2x3*3x2 is schoolbook (rank 9).
    
    For C12 and C22, we have 2x2*2x3 + 2x3*3x3 terms.
    2x2*2x3 is schoolbook (rank 12), 2x3*3x3 is schoolbook (rank 27).
    
    Total rank: 2*(7+9) + 2*(12+27) = 2*16 + 2*39 = 32 + 78 = 110. Worse than schoolbook!
    
    This approach isn't working. Let me reconsider.
    
    The knowledge says rank 92 is achievable. Let me think about what decomposition gives that.
    
    100 - 92 = 8 ranks saved. If each Strassen saves 1 rank (8->7), we need 8 Strassen applications.
    
    To get 8 instances of 2x2*2x2, we need to structure the problem to have 8 such subproblems.
    
    What if we split A (4x5) into 2x2, 2x2, 2x1 and B (5x5) into 2x5, 2x5, 1x5?
    
    A = [A1(2x2) | A2(2x2) | A3(2x1)]
    B = [B1(2x5); B2(2x5); B3(1x5)]
    
    C = A1*B1 + A2*B2 + A3*B3
    
    A1*B1: 2x2 * 2x5. We can further split B1 into B1a(2x2), B1b(2x3).
    A1*B1a: 2x2*2x2 -> Strassen (7)
    A1*B1b: 2x2*2x3 -> schoolbook (12)
    
    Similarly for A2*B2.
    
    A3*B3: 2x1*1x5 -> schoolbook (5)
    
    Total: 2*(7+12) + 5 = 2*19 + 5 = 43. Still not 92.
    
    I think I'm overcomplicating this. Let me just implement the schoolbook and accept rank 100, or try to find a better known decomposition.
    
    Actually, let me look up: the rank of 4x5*5x5 matrix multiplication.
    
    The schoolbook rank is 4*5*5 = 100.
    
    Known lower bounds and upper bounds... I believe the best known upper bound might be around 92-95.
    
    Let me try a different approach: use the fact that we can combine some terms.
    
    Actually, let me just implement the Strassen-based decomposition properly. The key is to identify all 2x2*2x2 subproblems and apply Strassen to each.
    
    For 4x5 * 5x5, if we view it as a sum of outer products, we have 100 terms. We can group them into 2x2*2x2 blocks where possible.
    
    Let me try: split the 4 rows of A into pairs, and the 5 columns of A / 5 rows of B into groups of 2 and 3.
    
    Rows of A: (0,1), (2,3)
    Cols of A / Rows of B: (0,1), (2,3), (4)
    
    This gives 2*3 = 6 blocks for A, and similarly for B.
    
    A blocks: A[0:2,0:2], A[0:2,2:4], A[0:2,4:5], A[2:4,0:2], A[2:4,2:4], A[2:4,4:5]
    B blocks: B[0:2,0:2], B[0:2,2:4], B[0:2,4:5], B[2:4,0:2], B[2:4,2:4], B[2:4,4:5]
    
    C blocks: C[0:2,0:2], C[0:2,2:4], C[0:2,4:5], C[2:4,0:2], C[2:4,2:4], C[2:4,4:5]
    
    C[i,j] = sum_k A[i,k]*B[k,j]
    
    C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2] + A[0:2,4:5]*B[4:5,0:2]
    = 2x2*2x2 + 2x2*2x2 + 2x1*1x2
    
    The first two are 2x2*2x2 (Strassen: 7 each), the last is 2x1*1x2 (schoolbook: 2).
    Total: 7+7+2 = 16
    
    Similarly for other blocks...
    
    This is getting very complex. Let me just implement the schoolbook for now and see if I can improve it.
    
    Actually, I realize I should just implement the known best result. Let me assume rank 92 is achievable and construct it.
    
    The construction would involve:
    1. Identifying 8 instances of 2x2*2x2 subproblems
    2. Applying Strassen to each (rank 7 instead of 8)
    3. Using schoolbook for the rest
    
    Total rank: 100 - 8 = 92.
    
    Let me code this up. I'll need to carefully track which tensor indices correspond to each subproblem.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook rank is n*m*p = 100
    # We'll try to reduce by using Strassen on 2x2*2x2 subblocks.
    
    # For now, let's implement the schoolbook and return it.
    # We can try to improve later.
    
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
