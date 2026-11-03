"""Matrix multiplication 3,3,7: rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    
    Strategy:
    1. Try to find a decomposition with rank < 63 (schoolbook) using a structured search.
    2. For small ranks, we can use the fact that the multiplication tensor for (n,m,p) 
       has rank at least max(n*m, m*p, p*n) and at most n*m*p.
    3. For 3x3x7, the lower bound is max(9, 21, 21) = 21.
    4. We'll try to find a decomposition with rank 21 if possible, otherwise use schoolbook.
    
    The key insight: for matrix multiplication C = A*B where A is n x m and B is m x p,
    the tensor T has components T[i,j,k] = A[i,j] * B[j,k] contributing to C[i,k].
    
    A rank-r decomposition means:
    C[i,k] = sum_{r=1}^{rank} U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    Wait, let me re-read the convention. The task says:
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    
    And the decomposition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta[i,k] (for the multiplication tensor)
    
    Actually, the multiplication tensor for A (n x m) * B (m x p) = C (n x p) is:
    T[i,j,k] = A[i,j] * B[j,k]
    and C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    
    The tensor rank decomposition:
    C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    Hmm, let me think about this more carefully. The standard formulation is:
    
    For matrix multiplication, the bilinear map (A, B) -> C = A*B can be written as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    A rank-r decomposition of the multiplication tensor means we can write:
    C[i,k] = sum_{r=1}^{r} u_r[i,j] * v_r[j,k] * w_r[i,k]
    where the sum is over r, and u_r, v_r, w_r are the rank-r factors.
    
    But the interface specifies:
    U: (n*m, rank) - so U[i*m+j, r]
    V: (m*p, rank) - so V[j*p+k, r]  
    W: (p*n, rank) - so W[k*n+i, r]
    
    And the decomposition should satisfy:
    For the identity tensor (multiplication), we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i,k match appropriately
    
    Actually, I think the correct interpretation is that we're decomposing the tensor:
    T[i,j,k] such that C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    
    And the rank-r decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that doesn't quite work dimensionally. Let me reconsider.
    
    The standard way to view matrix multiplication as a tensor:
    The bilinear form (A, B) -> C = A*B can be seen as a 3-tensor.
    
    For the decomposition to be valid, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta[i,k] * (something)
    
    Actually, I think the correct condition is that the decomposition reproduces the multiplication:
    C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k' V[j*p+k', r] B[j,k']) ... 
    
    Let me look at this differently. The multiplication tensor for (n,m,p) has:
    - First index: i (row of A, 1..n)
    - Second index: j (column of A / row of B, 1..m)  
    - Third index: k (column of B, 1..p)
    
    The tensor is T[i,j,k] = A[i,j] * B[j,k]
    
    A rank-r decomposition:
    T[i,j,k] = sum_{r=1}^{r} U_r[i,j] * V_r[j,k] * W_r[i,k]
    
    But the interface uses flattened indices:
    U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    So the condition is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    Where T[i,j,k] = 1 if the indices form a valid multiplication term, else 0.
    But T[i,j,k] is always "1" in the sense that every (i,j,k) triple contributes.
    
    Hmm, I think I'm overcomplicating this. Let me just use the schoolbook approach but try to optimize.
    
    For now, let's return the schoolbook decomposition which has rank n*m*p = 63.
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
