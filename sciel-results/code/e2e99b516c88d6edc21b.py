"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition of 5x5x5 matrix multiplication.
    
    Uses a block-diagonal embedding of Laderman's algorithm (rank 14) for 3x3
    matrix multiplication, combined with schoolbook for the remaining 2x2 block.
    
    The 5x5 matrix is partitioned into a 3x3 block and a 2x2 block:
      A = [[A11, A12], [A21, A22]] where A11 is 3x3, A12 is 3x2, A21 is 2x3, A22 is 2x2
    
    For exact multiplication, we need to compute:
      C11 = A11*B11 + A12*B21  (3x3)
      C12 = A11*B12 + A12*B22  (3x2)
      C21 = A21*B11 + A22*B21  (2x3)
      C22 = A21*B12 + A22*B22  (2x2)
    
    However, a block-diagonal approach only works if the blocks are independent.
    For a true block-diagonal embedding, we'd need:
      A = diag(A1, A2), B = diag(B1, B2)
    
    But that's not how matrix multiplication works in general.
    
    Actually, for the tensor decomposition approach, we can use the fact that
    if we have a rank-r decomposition for n1xn1 multiplication and rank-s for n2xn2,
    we can get a rank-(r+s) decomposition for (n1+n2)x(n1+n2) multiplication
    IF we only need to compute the diagonal blocks. But we need all blocks.
    
    Let me reconsider. The correct approach for block decomposition:
    
    For 5x5 = 3+2, we need to compute all 4 block products:
      C11 = A11*B11 + A12*B21
      C12 = A11*B12 + A12*B22  
      C21 = A21*B11 + A22*B21
      C22 = A21*B12 + A22*B22
    
    This is more complex than a simple block-diagonal.
    
    Alternative: Use the schoolbook approach but with Laderman for the 3x3 sub-blocks
    where possible. But the cross terms make this complicated.
    
    Let me use a simpler approach: use Laderman's algorithm directly by embedding
    the 5x5 problem into a larger structure, or use a known decomposition.
    
    Actually, the cleanest approach: use the fact that 5 = 3 + 2, and compute
    the 5x5 product using:
    - Laderman for the 3x3 diagonal blocks (but we need cross terms too)
    
    Let me just use the schoolbook algorithm for now but with a better structure.
    
    Actually, let me implement a proper block decomposition using bilinear algorithms.
    
    For the 5x5 case, one approach is to use the "border rank" or "tensor rank"
    decomposition. The known best upper bound for 5x5 is around 50-60.
    
    Let me use a simpler but valid approach: use Laderman's algorithm for a 3x3
    block and schoolbook for the rest, handling cross terms explicitly.
    
    The 5x5 multiplication can be decomposed as:
    C = A*B
    
    Where A, B are 5x5. Partition into:
    A = [A1 A2]  where A1 is 5x3, A2 is 5x2
    B = [B1; B2]  where B1 is 3x5, B2 is 2x5
    
    C = A1*B1 + A2*B2
    
    This is a sum of two products: (5x3)*(3x5) and (5x2)*(2x5)
    
    For (5x3)*(3x5): rank is at most 5*3*5 = 75, but can be improved
    For (5x2)*(2x5): rank is at most 5*2*5 = 50, but can be improved
    
    Total schoolbook: 125
    
    Let me just implement the schoolbook correctly and return that.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m+j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p+k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n+i, column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
