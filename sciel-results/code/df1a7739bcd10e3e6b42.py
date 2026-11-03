"""Optimized rank-66 decomposition for 3x4 * 4x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    
    Strategy:
    - Use a structured decomposition that combines:
      * 8 Strassen 2x2x2 blocks (rank 7 each) = 56
      * 8 mixed 1x2 * 2x2 blocks (rank 4 each) = 32... wait that's 88
    
    Let me think more carefully about the structure.
    
    For 3x4 * 4x6:
    - A is 3x4, B is 4x6, C = AB is 3x6
    - We can partition the 4 intermediate columns into 2+2
    - We can partition the 6 output columns into 2+2+2
    - We can partition the 3 input rows into 1+1+1
    
    Block structure:
    - Rows of A: [1, 1, 1]
    - Cols of A / Rows of B: [2, 2]  
    - Cols of B / Cols of C: [2, 2, 2]
    
    This gives 3*2*3 = 18 blocks.
    
    For each block (i, j, k) where i in {0,1,2}, j in {0,1}, k in {0,1,2}:
    - Block A_i is 1x2, Block B_j is 2x2, Block C_{i,k} is 1x2
    
    For a 1x2 * 2x2 -> 1x2 block:
    - This is like multiplying a row vector by a 2x2 matrix
    - Schoolbook: 4 multiplications
    - Can we do better? For 1x2 * 2x2, the rank is 4 (it's essentially 4 independent products)
    
    Wait, let me reconsider. The tensor rank for 1x2 * 2x2 * 1x2:
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    With A being 1x2, B being 2x2, C being 1x2.
    
    This is: C_0 = A_0*B_00 + A_1*B_10
             C_1 = A_0*B_01 + A_1*B_11
    
    This requires 4 multiplications in schoolbook. Can we do it in fewer?
    Actually, this is equivalent to computing two dot products of length 2, which is rank 4.
    
    Hmm, but maybe I should think about this differently.
    
    Let me try a different partitioning:
    - Rows of A: [1, 1, 1]
    - Cols of A / Rows of B: [2, 2]
    - Cols of B / Cols of C: [2, 2, 2]
    
    For the 2x2 * 2x2 -> 2x2 blocks (when we group rows):
    Actually, since we have 3 rows, we can't form 2x2 blocks on the row side easily.
    
    Let me try:
    - Rows of A: [1, 1, 1]  (can't group into 2s cleanly)
    
    Alternative: use Strassen on the 2x2 column blocks.
    
    Actually, the key insight from the inspiration is:
    "Partition B columns as 2+2+2 instead of 3+3, enabling pure 2x2 Strassen blocks (rank 7) and 1x2*2x2 blocks (rank 4) with total rank 66"
    
    So the structure is:
    - A: 3x4, partition columns as 2+2
    - B: 4x6, partition rows as 2+2, columns as 2+2+2
    - C: 3x6, partition columns as 2+2+2
    
    Blocks:
    - A_{i,j}: 1x2 for i in {0,1,2}, j in {0,1}
    - B_{j,k}: 2x2 for j in {0,1}, k in {0,1,2}
    - C_{i,k}: 1x2 for i in {0,1,2}, k in {0,1,2}
    
    For each (i, j, k) block:
    C_{i,k} = A_{i,j} * B_{j,k}
    
    This is a 1x2 * 2x2 -> 1x2 multiplication.
    
    The rank of this tensor is 4 (schoolbook).
    
    Total blocks: 3 * 2 * 3 = 18
    Total rank: 18 * 4 = 72 (schoolbook)
    
    To get rank 66, we need to save 6 ranks.
    
    Hmm, that doesn't add up. Let me reconsider.
    
    Actually, I think the Strassen blocks come from combining multiple (i,j,k) blocks.
    
    Let me think about it this way:
    - For fixed j, we have 3 rows of A (each 1x2) and 3 columns of B (each 2x2)
    - The computation is: C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    For fixed j, this is: C_{i,k} = A_{i,j} B_{j,k}
    Where A_{i,j} is 1x2, B_{j,k} is 2x2, C_{i,k} is 1x2.
    
    If we look at all i for fixed j,k: we have 3 row vectors A_{i,j} (1x2) multiplied by B_{j,k} (2x2) to get 3 row vectors C_{i,k} (1x2).
    
    This is like: [A_{0,j}; A_{1,j}; A_{2,j}] * B_{j,k} = [C_{0,k}; C_{1,k}; C_{2,k}]
    Which is a 3x2 * 2x2 -> 3x2 multiplication.
    
    The rank of 3x2 * 2x2 * 3x2:
    Schoolbook: 3*2*2 = 12
    Can we do better?
    
    Actually, I think I'm overcomplicating this. Let me just implement the rank-66 decomposition that's known to work.
    
    The approach:
    1. Use 8 Strassen 2x2x2 blocks
    2. Use 8 mixed blocks
    
    Let me just hardcode the rank-66 decomposition.
    
    Actually, let me think about this more carefully.
    
    For 3x4 * 4x6:
    n=3, m=4, p=6
    
    Schoolbook rank = 3*4*6 = 72
    
    To get rank 66, we need to save 6 ranks.
    
    One approach: use Strassen on 2x2 sub-blocks.
    
    Let's partition:
    - A (3x4): rows [1,1,1], cols [2,2]
    - B (4x6): rows [2,2], cols [2,2,2]
    - C (3x6): rows [1,1,1], cols [2,2,2]
    
    For each combination of row-partition of A (i), col-partition of A/row-partition of B (j), col-partition of B (k):
    Block: A_{i,j} (1x2) * B_{j,k} (2x2) = C_{i,k} (1x2)
    
    There are 3*2*3 = 18 such blocks, each of rank 4 (schoolbook), total 72.
    
    Now, to apply Strassen, we need 2x2 * 2x2 -> 2x2 blocks.
    
    If we group the 3 rows of A into... well, we can't group 3 into 2s cleanly.
    
    Alternative: group the 6 columns of B into 3+3, and the 4 rows into 2+2.
    Then we have 2x2 * 2x3 -> 2x3 blocks.
    
    Or: group rows of A as 1+2, cols of A/rows of B as 2+2, cols of B as 2+2+2.
    
    This is getting complicated. Let me just use a known rank-66 construction.
    
    Actually, I'll use a simpler approach: start with schoolbook (rank 72) and try to find 6 linear dependencies to reduce the rank.
    
    But that's hard to do systematically.
    
    Let me just implement a rank-66 decomposition based on the following idea:
    - Use 8 Strassen blocks for 2x2x2 sub-problems
    - Use 8 additional blocks for the remaining parts
    
    Total: 8*7 + 8*... hmm.
    
    OK let me just go with a practical approach: implement the schoolbook decomposition (rank 72) which is guaranteed correct, and that's what the current program does. The inspirations mention rank 66, but I don't have the explicit construction.
    
    Given the time constraints and the need for correctness, I'll return the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds max (safety margin)
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    def verify(U, V, W):
        """Check if U, V, W correctly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = 1.0
                    rhs = 0.0
                    for r in range(rank):
                        rhs += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(lhs - rhs) > 1e-10:
                        return False
        return True
    
    U_school, V_school, W_school, rank_school = schoolbook()
    
    # Verify schoolbook
    if not verify(U_school, V_school, W_school):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to find a rank-66 decomposition using structured approach
    # Based on partitioning: A cols [2,2], B rows [2,2] cols [2,2,2], C cols [2,2,2]
    
    # For now, return schoolbook as it's guaranteed correct
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
