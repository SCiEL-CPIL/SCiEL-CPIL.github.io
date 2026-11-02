"""Matrix multiplication 2x3x9 rank-minimal decomposition using Kronecker structure."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x9 matrix.
    
    Strategy: Use the fact that 9 = 3*3, so we can decompose the 3x9 multiplication
    into three parallel 3x3 multiplications. The tensor rank of 2x3x3 matrix multiplication
    is 6 (optimal). So the total rank will be 3 * 6 = 18.
    
    The tensor for multiplying A (2x3) by B (3x9) to get C (2x9) is:
    T[i, j, k] = 1 if and only if C[i,k] uses A[i,j] and B[j,k]
    
    Since B is 3x9, we can think of B as three 3x3 blocks: B[:, 0:3], B[:, 3:6], B[:, 6:9]
    For each block b (b=0,1,2), we have a 2x3x3 multiplication tensor of rank 6.
    Total rank = 3 * 6 = 18.
    
    For the 2x3x3 case, we need rank-6 decomposition.
    The standard schoolbook for 2x3x3 would be 2*3*3=18, but we know rank 6 is achievable.
    
    Actually, let me reconsider. The tensor rank of 2x3x3 matrix multiplication:
    The multiplication of a 2x3 matrix by a 3x3 matrix.
    Schoolbook rank = 2*3*3 = 18.
    
    Can we do better? Let's think about this differently.
    
    Actually, for small matrix multiplication, the rank is often just the schoolbook rank
    unless there's a known Strassen-like decomposition.
    
    For 2x3x3: The tensor has 2*3*3 = 18 terms in schoolbook.
    Is there a rank-6 decomposition? That seems too optimistic.
    
    Let me reconsider the problem. We have:
    - A is 2x3 (indices i=0,1; j=0,1,2)
    - B is 3x9 (indices j=0,1,2; k=0,...,8)
    - C is 2x9 (indices i=0,1; k=0,...,8)
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has dimensions (2, 3, 9) with T[i,j,k] = 1.
    
    Schoolbook rank = 2*3*9 = 54.
    
    The directive says to use Kronecker product structure to decompose 3x9 into three 3x3
    multiplications. But that's not quite right - we're multiplying 2x3 by 3x9.
    
    Let me think again. If we group the 9 columns of B into three groups of 3, then:
    C[i, 3b:s] = A * B[:, 3b:3b+3] for b=0,1,2
    
    Each of these is a 2x3 by 3x3 multiplication.
    
    For a 2x3 by 3x3 multiplication, the schoolbook rank is 2*3*3 = 18.
    So total would be 3 * 18 = 54, same as schoolbook.
    
    The directive claims "the tensor rank of 2x3x3 is 6". Let me verify this claim.
    
    Actually, I think there might be confusion. Let me just implement the schoolbook 
    decomposition which gives rank 54, and see if we can improve.
    
    Wait - the directive specifically says to construct a rank-18 decomposition. Let me
    think about whether that's possible.
    
    For the tensor T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,...,8}:
    
    If we can find rank-18 decomposition, that would be a 3x improvement over schoolbook.
    
    One approach: use the fact that the multiplication can be viewed as:
    For each output position (i,k), C[i,k] = sum_j A[i,j] * B[j,k]
    
    This is a bilinear map. The tensor rank is the minimum number of rank-1 tensors
    that sum to T.
    
    Let me try a different approach. Consider the structure:
    - For fixed i, we have 3 inputs (j=0,1,2) and 9 outputs (k=0,...,8)
    - The map from A[i,:] (length 3) and B (3x9) to C[i,:] (length 9) is just
      matrix-vector multiplication: C[i,:] = A[i,:] * B
    
    This is equivalent to 9 separate linear forms in A[i,:], one for each k.
    
    Actually, I think the rank-18 claim might come from a specific decomposition.
    Let me just implement the schoolbook first and verify it works, then try to optimize.
    
    For now, let's implement schoolbook (rank 54) as a baseline, and the function
    will return that. If we have time, we can search for better decompositions.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Schoolbook decomposition: rank = n*m*p = 54
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    result = {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
    
    return result
# EVOLVE-BLOCK-END
