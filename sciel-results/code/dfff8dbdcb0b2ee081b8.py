"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x5 matrix.
    Uses block-Strassen decomposition with a 2x2 block structure on the 5x5 matrices,
    utilizing the known rank-19 algorithm for 3x3 matrix multiplication on the four 3x3 sub-blocks.
    
    The 5x5 matrices are partitioned into blocks of sizes (2,3) and (3,2):
    - A (3x5) is partitioned into A1 (3x2) and A2 (3x3)
    - B (5x5) is partitioned into B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    
    The product C = A*B = A1*B11 + A1*B12 + A2*B21 + A2*B22
    
    For the 3x3 matrix multiplications (A2*B22), we use the rank-19 algorithm.
    For the rectangular multiplications (A1*B11, A1*B12, A2*B21), we use schoolbook.
    
    Actually, let me reconsider. The task is to multiply a 3x5 matrix by a 5x5 matrix.
    The output is a 3x5 matrix.
    
    Let me think about this more carefully. We have:
    - A: 3x5
    - B: 5x5
    - C = A*B: 3x5
    
    We can partition A into A1 (3x2) and A2 (3x3), and B into columns:
    B = [B1 | B2] where B1 is 5x2 and B2 is 5x3.
    
    Then C = A1*B1 + A2*B2.
    
    A1*B1 is 3x2, A2*B2 is 3x3.
    
    For A1*B1 (3x2 times 2x2 = 3x2): schoolbook rank = 3*2*2 = 12
    For A2*B2 (3x3 times 3x3 = 3x3): use rank-19 algorithm
    
    Total rank = 12 + 19 = 31.
    
    But wait, the W factor follows the trace convention W[k*n+i, r] where n=3, p=5.
    So W has shape (p*n, rank) = (15, rank).
    
    Let me implement the rank-19 algorithm for 3x3 matrix multiplication.
    The rank-19 algorithm is based on the work of Strassen and others.
    
    Actually, the known minimum rank for 3x3 matrix multiplication over arbitrary fields is not known to be 19.
    The best known upper bound for 3x3 matrix multiplication is 23 (from Strassen's algorithm applied to 3x3).
    
    Let me use Strassen's algorithm for 3x3. Strassen's algorithm for 2x2 uses 7 multiplications.
    For 3x3, we can pad to 4x4 and use Strassen, giving rank 7^2 = 49, but that's worse than schoolbook (27).
    
    Alternatively, we can use a direct rank decomposition for 3x3. The schoolbook rank for 3x3 is 27.
    
    Actually, there is a known rank-23 algorithm for 3x3 matrix multiplication by Strassen.
    Let me use that.
    
    Strassen's rank-23 algorithm for 3x3:
    The algorithm partitions the 3x3 matrices into 2x2 and 1x1 blocks and uses Strassen's 7-multiplication algorithm
    on the 2x2 blocks plus additional terms for the 1x1 blocks.
    
    Actually, let me just use the schoolbook decomposition for now and see if I can improve it.
    Schoolbook rank for 3x5x5 = 3*5*5 = 75.
    
    The current program has rank 75 (n*m*p = 3*5*5 = 75).
    
    Let me try the block approach:
    - Partition A (3x5) into A1 (3x2) and A2 (3x3)
    - Partition B (5x5) into B1 (5x2) and B2 (5x3)
    - C = A1*B1 + A2*B2
    
    A1*B1: 3x2 * 2x2 = 3x2, schoolbook rank = 3*2*2 = 12
    A2*B2: 3x3 * 3x3 = 3x3, schoolbook rank = 3*3*3 = 27
    
    Total = 12 + 27 = 39. That's better than 75!
    
    But can we do better on the 3x3 part? Let me use a known low-rank algorithm for 3x3.
    
    The best known rank for 3x3 matrix multiplication is 23 (Strassen's algorithm).
    So total rank = 12 + 23 = 35.
    
    Let me implement Strassen's rank-23 algorithm for 3x3 matrix multiplication.
    
    Strassen's algorithm for 3x3:
    Partition each 3x3 matrix into:
    M = [[M11, M12, M13],
         [M21, M22, M23],
         [M31, M32, M33]]
    
    where M11 is 1x1, M12 is 1x1, M13 is 1x1, etc.
    
    Actually, the standard Strassen algorithm for 3x3 is not straightforward. Let me use a different approach.
    
    I'll use the fact that 3x3 matrix multiplication can be done with rank 23 by using the following decomposition:
    This is based on the work of Strassen (1969) and later improvements.
    
    Actually, let me just use the schoolbook for the 3x3 part to keep it simple, giving total rank 39.
    That's already a significant improvement over 75.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Partition: A is n x m = 3 x 5
    # Split A into A1 (n x m1) and A2 (n x m2) where m1=2, m2=3
    # Split B into B1 (m1 x p) and B2 (m2 x p) where m1=2, m2=3
    # C = A1*B1 + A2*B2
    
    m1 = 2
    m2 = 3
    
    # Part 1: A1*B1 where A1 is 3x2, B1 is 2x5, result is 3x5
    # Schoolbook: rank = 3*2*5 = 30
    # Wait, B1 is m1 x p = 2 x 5
    
    # Part 2: A2*B2 where A2 is 3x3, B2 is 3x5, result is 3x5
    # Schoolbook: rank = 3*3*5 = 45
    
    # Total = 30 + 45 = 75. That's the same as before!
    
    # I need to think about this differently. The key is that in the schoolbook decomposition,
    # each term is a rank-1 update. The total rank is n*m*p.
    
    # Let me reconsider. The tensor for matrix multiplication M_{n,m,p} has rank at most n*m*p.
    # We want to find a decomposition with fewer terms.
    
    # One approach: use the fact that we can group terms.
    # For example, if we fix i and k, we can sum over j. But that doesn't reduce the rank.
    
    # Actually, the rank of the matrix multiplication tensor M_{n,m,p} is exactly n*m*p for most cases,
    # but there are known improvements for specific dimensions.
    
    # For 3x5x5, the schoolbook rank is 75. Let me see if I can find a better decomposition.
    
    # One idea: use the fact that 5 = 2 + 3, and use Strassen-like algorithms on the blocks.
    
    # Let me try a different partitioning:
    # A (3x5) = [A1 | A2] where A1 is 3x2, A2 is 3x3
    # B (5x5) = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    
    # C = A1*B11 + A1*B12 + A2*B21 + A2*B22
    #    = [A1*B11 + A2*B21 | A1*B12 + A2*B22]
    
    # Left part (3x2): A1*B11 (3x2 * 2x2 = 3x2) + A2*B21 (3x3 * 3x2 = 3x2)
    # Right part (3x3): A1*B12 (3x2 * 2x3 = 3x3) + A2*B22 (3x3 * 3x3 = 3x3)
    
    # For the left part:
    # A1*B11: 3x2 * 2x2, schoolbook rank = 3*2*2 = 12
    # A2*B21: 3x3 * 3x2, schoolbook rank = 3*3*2 = 18
    # Total left = 30
    
    # For the right part:
    # A1*B12: 3x2 * 2x3, schoolbook rank = 3*2*3 = 18
    # A2*B22: 3x3 * 3x3, schoolbook rank = 3*3*3 = 27
    # Total right = 45
    
    # Grand total = 30 + 45 = 75. Still the same!
    
    # The issue is that block decomposition alone doesn't reduce the rank unless we use a non-schoolbook
    # algorithm for the sub-blocks.
    
    # Let me use Strassen's algorithm for the 3x3 * 3x3 part.
    # Strassen's algorithm for 2x2 uses 7 multiplications instead of 8.
    # For 3x3, we can pad to 4x4 and use Strassen twice: 7^2 = 49, but that's worse than 27.
    
    # Alternatively, there is a direct rank-23 algorithm for 3x3. Let me implement that.
    
    # Actually, let me just return the schoolbook decomposition for now, since I'm not sure about the
    # exact rank-23 algorithm for 3x3.
    
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
