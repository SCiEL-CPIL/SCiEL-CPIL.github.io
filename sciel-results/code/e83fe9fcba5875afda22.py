"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a low-rank decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy: Use the standard schoolbook decomposition but try to find
    a lower rank by exploiting structure. For 4x5*5x5, the schoolbook rank is 100.
    
    However, we can potentially do better by recognizing that the 4x5 matrix
    multiplication can be viewed as 4 independent 1x5 multiplications, each
    requiring 5 scalar multiplications, giving 20 rank-1 terms for the "diagonal"
    part. But that's not quite right because the output is 4x5.
    
    Actually, let me reconsider. The tensor T[i,j,k] = delta_{j,k} for (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    Wait, no. The matrix multiplication tensor is T_{i,j,k} where the output (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    So the tensor is T[i,j,k] = 1 if we're looking at the term A[i,j]*B[j,k].
    
    The standard decomposition has rank n*m*p = 4*5*5 = 100.
    
    Can we do better? For square matrices, Strassen gives rank 7 for 2x2.
    For 4x4, we could use Strassen twice to get rank 7^2 = 49, but that's for 4x4*4x4.
    
    For 4x5*5x5, this is more complex. Let's think about it differently.
    
    Actually, let's just implement the schoolbook decomposition first and see if we can improve.
    
    The key insight: we need to find rank-1 tensors that sum to the matrix multiplication tensor.
    A rank-1 tensor is U[i,j] * V[j,k] * W[k,i] (with appropriate index mapping).
    
    Let me use the explicit index mapping:
    - U has shape (n*m, rank) = (20, rank)
    - V has shape (m*p, rank) = (25, rank)  
    - W has shape (p*n, rank) = (20, rank)
    
    The tensor entry T[i, j, k] for i in [0,n), j in [0,m), k in [0,p)
    is decomposed as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And we need this to equal delta_{j,k} (since (AB)_{i,k} = sum_j A_{i,j} B_{j,k}).
    
    Wait, that's not right either. Let me re-read the problem.
    
    "Matrix multiplication 4,5,5" - multiply a 4x5 matrix by a 5x5 matrix.
    Output is 4x5.
    
    (AB)_{i,k} = sum_{j=0}^{4} A_{i,j} * B_{j,k}
    
    So the tensor T[i, j, k] = 1 for all valid i, j, k, and the decomposition must satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    
    Wait, that's not right. The tensor should be such that:
    T[i, j, k] = 1 if the term A_{i,j} * B_{j,k} contributes to (AB)_{i,k}.
    
    Actually, I think the tensor is just T[i,j,k] = 1 for all i,j,k, and the decomposition
    must reproduce the bilinear form. Let me think again.
    
    The matrix multiplication can be written as:
    (AB)_{i,k} = sum_{j} A_{i,j} B_{j,k}
    
    This is a trilinear form. The tensor T is defined by T[i,j,k] = 1, and we want to write
    T = sum_c u^{(c)} v^{(c)} w^{(c)} where u^{(c)} is a vector in R^{n*m}, etc.
    
    No wait, I think I'm overcomplicating this. Let me look at the seed program again.
    
    In the seed:
    U[i*m+j][column] = 1
    V[j*p+k][column] = 1  
    W[k*n+i][column] = 1
    
    So for each (i,j,k), we have a rank-1 term where U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1.
    
    This means the decomposition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_{(i',j',k') where c corresponds to (i',j',k')} [i==i'] * [j==j'] * [k==k']
    
    Wait, that gives 1 only when i==i' AND j==j' AND k==k', which is just the identity in the (i,j,k) space.
    
    But we need sum_j A_{i,j} B_{j,k}, which is different.
    
    I think I'm misunderstanding the tensor structure. Let me re-read the problem statement.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 5 matrix."
    
    The matrix multiplication tensor for n x m * m x p is a tensor T in R^{n*m} x R^{m*p} x R^{p*n} such that:
    T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i for some i,j,k, else 0.
    
    Wait, that's what the seed does. But that doesn't seem right for matrix multiplication.
    
    Let me think about this more carefully. The output (AB)_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    We can write this as:
    (AB)_{i,k} = sum_{j} A_{i,j} * B_{j,k} * 1
    
    So the "tensor" is T[i,j,k] = 1 for all i,j,k, and the decomposition must satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k?
    
    No, that's not right either. The issue is that the same A_{i,j} and B_{j,k} are used for different outputs.
    
    I think the correct formulation is:
    We want to compute C = A * B where C[i,k] = sum_j A[i,j] * B[j,k].
    
    This can be written as a sum of rank-1 terms:
    C[i,k] = sum_c (sum_{(i',j') in S_c} U[i*m+j', c] * A[i',j']) * (sum_{(j'',k') in T_c} V[j''*p+k', c] * B[j'',k']) * W[k*n+i, c]
    
    Hmm, this is getting complicated. Let me just trust the seed program's formulation and try to improve it.
    
    The seed uses rank = n*m*p = 100. Can we do better?
    
    For now, let's just return the schoolbook decomposition. The task is to find a rank-minimal decomposition, and 100 is the schoolbook rank.
    
    Actually, I realize I should just implement the schoolbook decomposition correctly and return it. The score is the rank, so lower is better. 100 is the baseline.
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
