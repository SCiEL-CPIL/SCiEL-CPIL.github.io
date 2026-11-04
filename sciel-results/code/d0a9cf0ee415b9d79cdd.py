"""Matrix multiplication 2x4 by 4x8 - rank-minimal tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Strategy: Use a block-based approach. Treat the 2x4 matrix as having 2 rows,
    each of length 4. The 4x8 matrix has 4 rows of length 8.
    
    We can think of this as: for each output element C[i,k] = sum_j A[i,j] * B[j,k].
    
    A simple but effective approach: use the fact that we can group columns.
    
    Actually, let's think about this more carefully. The schoolbook is rank 64.
    
    Key insight: We can use a Kronecker product structure. 
    2x4 * 4x8 = (2x1 * 1x4) * (4x1 * 1x8) in some sense?
    
    Better approach: Think of A as a 2x4 matrix. We can write:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    One approach: use the fact that 4 = 2*2 and 8 = 2*4 or 8 = 4*2.
    
    Let's try: split the 4 middle dimension into two groups of 2.
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x8
    
    C = A1*B1 + A2*B2
    
    Each 2x2 * 2x8 multiplication can be done with rank 2*2*8 = 32 schoolbook,
    but we can do better. Actually 2x2 * 2x8: for each pair (i,j) in the 2x2 part
    and each k in the 8, we need A[i,j]*B[j,k]. That's 2*2*8 = 32 terms per block.
    Two blocks = 64. Same as schoolbook.
    
    Alternative: use Strassen-like ideas. For 2x2 * 2x8:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    We can write this as: for each k, C[:,k] = A * B[:,k] where A is 2x2 and B[:,k] is 2x1.
    2x2 * 2x1 multiplication has rank 4 (schoolbook) or can we do better?
    Actually 2x2*2x1: C1 = a11*b1 + a12*b2, C2 = a21*b1 + a22*b2. Rank 4 schoolbook.
    This is just a bilinear map from (2x2, 2x1) to (2x1). The tensor is in R^4 x R^2 x R^2.
    The border rank is at most 4, and I believe the rank is 4.
    
    So 2x2*2x8 = 8 copies of 2x2*2x1, each rank 4, total rank 32. Two blocks = 64.
    
    Let me think differently. What if we use the structure that 4=2+2 and combine?
    
    C = A1*B1 + A2*B2
    
    Using Strassen for the 2x2 * 2x2 part (ignoring the 8 for a moment):
    A1*B1 + A2*B2 where each is 2x2*2x2... but B1 and B2 are 2x8, not 2x2.
    
    Let me try another decomposition. Write the problem as:
    A is 2x4, B is 4x8.
    
    Think of A as a vector of 8 elements, B as a vector of 32 elements.
    C is a vector of 16 elements.
    
    The tensor T is in R^8 x R^32 x R^16.
    
    Schoolbook: rank 64.
    
    Can we do better? The known lower bounds for matrix multiplication rank are weak.
    For 2x4*4x8, I'm not sure of the exact minimal rank.
    
    Let me try a practical approach: use a mixed decomposition.
    
    Approach: Split the 4 middle dimension as 4 = 2 + 2.
    A = [A_left | A_right], each 2x2
    B = [B_top; B_bottom], each 2x8
    
    C = A_left * B_top + A_right * B_bottom
    
    For each 2x2 * 2x8 product, use the following:
    Let X be 2x2, Y be 2x8. Z = X*Y, Z is 2x8.
    Z[i,k] = X[i,0]*Y[0,k] + X[i,1]*Y[1,k]
    
    For each k (8 values), this is a 2x2*2x1 multiply, rank 4.
    Total for one block: 8*4 = 32. Two blocks: 64.
    
    Alternative for 2x2*2x8: 
    Z[0,k] = X[0,0]*Y[0,k] + X[0,1]*Y[1,k]
    Z[1,k] = X[1,0]*Y[0,k] + X[1,1]*Y[1,k]
    
    We can define:
    P0 = X[0,0]*Y[0,k] for all k -> vector of length 8
    P1 = X[0,1]*Y[1,k] for all k -> vector of length 8
    P2 = X[1,0]*Y[0,k] for all k
    P3 = X[1,1]*Y[1,k] for all k
    
    Z[0,k] = P0[k] + P1[k]
    Z[1,k] = P2[k] + P3[k]
    
    This is still rank 32 (4 vectors of length 8).
    
    Hmm, can we share computations? 
    Let's define:
    S0 = X[0,0] + X[0,1]
    S1 = X[1,0] + X[1,1]
    
    Then:
    Z[0,k] = X[0,0]*Y[0,k] + X[0,1]*Y[1,k]
    = (S0/2 + (X[0,0]-X[0,1])/2) * Y[0,k] + (S0/2 - (X[0,0]-X[0,1])/2) * Y[1,k]
    This doesn't obviously help.
    
    Let me try yet another approach. What if we use the fact that we only need
    integer/half-integer entries?
    
    Actually, let me just try to implement a reasonable decomposition and see.
    
    The best I can think of quickly is the block approach which gives 64 (same as schoolbook).
    Let me try to find something better.
    
    Idea: Use a 2x2 Strassen-like decomposition on the "outer" structure.
    
    Write A (2x4) as a single row of 4 elements repeated? No, that doesn't work.
    
    Let me try: C[i,k] = sum_j A[i,j] B[j,k]
    
    Define for the 2x4 * 4x8 case:
    
    Use 7 multiplications via a clever grouping?
    
    Actually, for 2x4*4x8, one approach:
    - Group the 4 middle indices into pairs: (0,1) and (2,3)
    - A = [A0 | A1] where A0, A1 are 2x2
    - B = [B0; B1] where B0, B1 are 2x8
    - C = A0*B0 + A1*B1
    
    For the 2x2*2x8 part, I'll use rank 32 (schoolbook for that sub-problem).
    Total: 64.
    
    Let me just go with a clean implementation. If I can't beat 64 easily, at least
    I'll have a correct answer. But the score is 1.0 for rank 64, so I need to do better.
    
    Wait - the score is seed/value where seed is the schoolbook rank (64) and value is our rank.
    Score 1.0 means value=64. To improve, I need value < 64.
    
    Let me think harder. Is there a known decomposition for 2x4*4x8 with rank < 64?
    
    One idea: use the fact that 4 = 2^2 and 8 = 2^3, so this is related to tensor powers.
    The 2x2*2x2 matrix multiplication has rank at most 7 (Strassen).
    
    Can we express 2x4*4x8 in terms of 2x2*2x2 multiplications?
    
    2x4*4x8: Think of A as 2x(2x2) and B as (2x2)x8.
    Reshape A to 2x2x2 and B to 2x2x4 (since 4x8 = 2x2x4).
    
    C[i,k] = sum_{j} A[i,j] B[j,k]
    = sum_{j1,j2} A[i,j1,j2] B[j1,j2,k]
    
    where j = j1*2 + j2 (or similar indexing).
    
    This is a contraction over a 2x2 index. If we could use Strassen for the 2x2 "matrix" part...
    but it's not quite a matrix multiplication because the dimensions don't match up that way.
    
    Actually: C[i,k] = sum_{j1,j2} A[i,j1,j2] B[j1,j2,k]
    
    This is like: for each (i,k), we're computing a dot product of A[i,:,:] and B[:,:,k] over the 2x2 grid.
    
    If we fix i and k, this is just a sum of 4 products. The "matrix multiplication" interpretation
    would require that the free indices form matrices, which they do: i is 2, k is 8, and the
    contracted index is 4 (which is 2x2).
    
    This is exactly a 2x4 * 4x8 matrix multiplication, which is what we started with.
    
    Let me try a different tactic: just implement a rank-64 solution cleanly and return it.
    The score will be 1.0, same as the seed. But maybe I can find a small improvement.
    
    Actually, wait. Let me reconsider. The task says "rank-minimal exact tensor decomposition."
    The score is seed/value = 64/rank. So if I find rank 56, score = 64/56 = 1.143.
    
    From the observations, a block-Strassen approach achieved rank 56. Let me implement that.
    
    Block-Strassen for 2x4*4x8:
    Split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each block is 1x2 (since 2 rows / 2 = 1, 4 cols / 2 = 2)
    Wait, 2x4 split into 2x2 blocks: each block is 1x2.
    B = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each block is 2x4 (4 rows/2=2, 8 cols/2=4)
    
    Hmm, the blocks aren't square. Let me reconsider.
    
    A is 2x4. Split rows into 2 groups of 1, columns into 2 groups of 2:
    A00 (1x2), A01 (1x2), A10 (1x2), A11 (1x2)
    
    B is 4x8. Split rows into 2 groups of 2, columns into 2 groups of 4:
    B00 (2x4), B01 (2x4), B10 (2x4), B11 (2x4)
    
    C = A * B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
                 [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Each block of C is 1x4.
    
    C00 = A00*B00 + A01*B10  (1x2 * 2x4 + 1x2 * 2x4 = 1x4)
    
    For 1x2 * 2x4: that's just a dot product of a 2-vector with each column of the 2x4 matrix.
    Rank: 2*4 = 8 schoolbook. Or can we do better? 1x2*2x4: C[k] = a0*B[0,k] + a1*B[1,k] for k=0..3.
    This is rank 8 (each term a_i*B[j,k] is one rank-1 term). I don't think we can do better than 8
    for this specific shape since it's essentially 4 separate 1x2*2x1 products.
    
    So C00 = A00*B00 + A01*B10 requires 8+8=16 multiplications.
    Similarly for other blocks: 4 blocks * 16 = 64. Same as schoolbook.
    
    This isn't helping. Let me think about the observation that said "block-Strassen approach using
    2x2 sub-blocks, reducing rank to 56."
    
    Maybe the approach is different. Perhaps they're treating the problem as a 2x2 matrix
    multiplication where each "element" is itself a matrix, and using Strassen's 7 multiplications.
    
    For that to work, we need the sub-matrices to be square or at least compatible.
    
    Let me try: view 2x4*4x8 as follows.
    A is 2x4. Think of it as a 2x1 matrix of 1x4 vectors.
    B is 4x8. Think of it as a 4x8 matrix.
    
    Alternatively: A is 2x4, B is 4x8. 
    Write A = A1 * A2 where A1 is 2x2 and A2 is 2x4? No, that changes the problem.
    
    OK here's another idea. Use the identity:
    2x4 * 4x8 = (2x2 * 2x4) * (2x4 * 4x8)? No, dimensions don't work.
    
    Let me just try to implement a rank-56 solution based on the hint.
    
    Perhaps the approach is:
    - Use 7 Strassen multiplications for a 2x2 "outer" structure
    - But the inner dimensions are 4 and 8
    
    Specifically, treat the problem as:
    C = A * B where A is 2x4, B is 4x8.
    
    Split A into two 2x2 blocks: A = [A_L | A_R] (each 2x2)
    Split B into two 2x8 blocks: B = [B_T; B_B] (each 2x8)
    
    C = A_L * B_T + A_R * B_B
    
    Now, A_L * B_T is 2x2 * 2x8, and A_R * B_B is 2x2 * 2x8.
    
    For 2x2 * 2x8:
    Let X be 2x2, Y be 2x8. Z = X*Y is 2x8.
    
    Using Strassen-like ideas on the 2x2 part:
    Z[0,k] = X[0,0]*Y[0,k] + X[0,1]*Y[1,k]
    Z[1,k] = X[1,0]*Y[0,k] + X[1,1]*Y[1,k]
    
    Standard: 4 multiplications per k, 8 k's = 32.
    
    Strassen for 2x2*2x1 (per column k):
    M1 = (X[0,0]+X[1,1]) * (Y[0,k]+Y[1,k])
    M2 = (X[1,0]+X[1,1]) * Y[0,k]
    M3 = X[0,0] * Y[1,k]
    M4 = X[1,1] * (Y[1,k] - Y[0,k])
    
    Z[0,k] = M1 - M4 + M3
    Z[1,k] = M2 + M4
    
    This uses 4 multiplications per column, same as schoolbook for 2x2*2x1.
    Strassen doesn't help for 2x2*2x1 because it's already minimal at rank 4.
    
    Hmm. So 2x2*2x8 has rank 32 (8 columns * 4 per column).
    Two such blocks: 64.
    
    I'm stuck at 64. Let me just implement the schoolbook cleanly and return rank 64.
    
    Actually, you know what, let me just go with the schoolbook. The score will be 1.0,
    same as the seed. At least it's correct.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = n * m * p  # 64
    
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
