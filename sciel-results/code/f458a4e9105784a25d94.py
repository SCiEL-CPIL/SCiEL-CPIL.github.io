"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    We use the Kronecker product structure:
    - 4x4 * 4x7 = (2x2 ⊗ 2x2) * (2x2 ⊗ 2x4)
    - This reduces to rank(r_222) * rank(2x2x2) * ... but we need to be careful.
    
    Actually, let's think more carefully. The tensor for (n,m,p) matrix multiplication
    can be decomposed using Kronecker products when the dimensions are composite.
    
    For 4x4x7: We can write 4 = 2*2, 4 = 2*2, 7 = 7*1 (prime).
    
    The schoolbook rank is 4*4*7 = 112.
    
    Let's try a different approach. We know that:
    - 2x2x2 has rank 7 (Strassen)
    - 2x1x1 has rank 2 (trivial: just two multiplications)
    - 1x1x1 has rank 1
    
    Actually, for the tensor decomposition of matrix multiplication:
    If we have a decomposition for (n1,m1,p1) of rank r1 and (n2,m2,p2) of rank r2,
    we can construct a decomposition for (n1*n2, m1*m2, p1*p2) of rank r1*r2.
    
    So for 4x4x7:
    - We could use 2x2x2 (rank 7) ⊗ 2x2x1 (rank ?)
    - Or 2x2x7 (rank ?) ⊗ 2x2x1 (rank ?)
    
    Let's think about what ranks are known:
    - 2x2x2: rank 7 (Strassen)
    - 2x2x1: This is multiplying 2x2 by 2x1. Schoolbook is 2*2*1=4.
      Can we do better? The tensor is sum_{i,j} e_i ⊗ e_j ⊗ (e_i ⊗ e_j in output?)
      Actually for (2,2,1): A is 2x2, B is 2x1, C is 2x1.
      C[0] = A[0,0]*B[0] + A[0,1]*B[1]
      C[1] = A[1,0]*B[0] + A[1,1]*B[1]
      This requires at least 4 multiplications since each output entry depends on 2 products,
      and there's no sharing. So rank = 4.
      
    - 2x2x7: A is 2x2, B is 2x7, C is 2x7.
      C[:,k] = A @ B[:,k] for each k.
      Each column of C is a 2x2 times 2x1, which takes 4 multiplications.
      With 7 columns, schoolbook is 28.
      But can we share across columns? The A matrix is shared.
      Using Strassen for each column: 7 multiplications per column, total 49. That's worse.
      Actually Strassen reduces 8 to 7 for 2x2x2. For 2x2x7, we compute 2x2 * 2x7.
      We can view this as: for each of the 7 columns, we do a 2x2x2 multiplication.
      Using Strassen: 7 multiplications per column, so 49 total.
      Schoolbook: 4 multiplications per column, so 28 total.
      So schoolbook is better here. Rank = 28.
      
    So 4x4x7 via (2x2x2) ⊗ (2x2x7): rank = 7 * 28 = 196. That's worse than 112.
    
    Via (2x2x7) ⊗ (2x2x2): same thing, 28 * 7 = 196.
    
    What about 4x4x7 via (4x4x1) ⊗ (1x1x7)?
    - 4x4x1: A is 4x4, B is 4x1, C is 4x1. Schoolbook: 16.
    - 1x1x7: A is 1x1, B is 1x7, C is 1x7. Schoolbook: 7.
    - Total: 16 * 7 = 112. Same as schoolbook.
    
    Via (4x1x7) ⊗ (1x4x1):
    - 4x1x7: A is 4x1, B is 1x7, C is 4x7. Schoolbook: 4*1*7=28.
    - 1x4x1: A is 1x4, B is 4x1, C is 1x1. Schoolbook: 4.
    - Total: 28 * 4 = 112.
    
    Via (2x4x7) ⊗ (2x1x1):
    - 2x4x7: A is 2x4, B is 4x7, C is 2x7. Schoolbook: 2*4*7=56.
    - 2x1x1: A is 2x1, B is 1x1, C is 2x1. Schoolbook: 2.
    - Total: 56 * 2 = 112.
    
    Hmm, all these give 112. The Kronecker product approach doesn't help when one factor is prime.
    
    Let me reconsider. Maybe we can use a mixed approach.
    
    Actually, for 4x4x7, let's think of it as computing C = A*B where A is 4x4 and B is 4x7.
    We can partition A and B into 2x2 blocks.
    
    A = [[A11, A12], [A21, A22]] where each is 2x2
    B = [[B11, B12, B13, B14, B15, B16, B17],
         [B21, B22, B23, B24, B25, B26, B27]]
    where B1j and B2j are 2x1 vectors (column j of the top/bottom half).
    
    C = [[A11*B1j + A12*B2j for j=1..7],
         [A21*B1j + A22*B2j for j=1..7]]
    
    Each 2x2 * 2x1 multiplication takes 4 multiplications (schoolbook) or we can try to be clever.
    
    Total schoolbook: 4 blocks * 7 columns * 4 mults = 112.
    
    I think for this specific problem with 7 being prime, the schoolbook rank of 112 might be optimal,
    or very close to it. Let me just return the schoolbook decomposition.
    
    Actually, wait. Let me reconsider the problem. The task says "rank-minimal exact tensor decomposition."
    The score is the rank, lower is better. The seed program has rank 112 (schoolbook).
    
    Let me check if there's a known better algorithm for 4x4x7.
    
    Actually, I realize I should just implement the schoolbook decomposition correctly and return it.
    The current program already does this. Let me verify it's correct and maybe optimize the code.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 4 * 4 * 7 = 112
    
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
