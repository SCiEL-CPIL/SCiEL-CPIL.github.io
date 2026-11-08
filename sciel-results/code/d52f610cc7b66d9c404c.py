"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    The multiplication tensor has dimensions (n, m, p) = (2, 3, 6).
    We need to find rank r and matrices U (n*m x r), V (m*p x r), W (p*n x r)
    such that the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] = 1 (it's the identity-like tensor for matrix mult).
    Wait, the multiplication tensor T_{ijk} satisfies:
    sum_{j} A_{ij} B_{jk} = C_{ik}
    This is equivalent to: sum_{j,c} U_{ij,c} V_{jc,c} W_{kc,c} = A_{ik}
    
    Actually, the standard formulation: we want to compute all products A[i,j]*B[j,k]
    The tensor is the 3-tensor where T[i,j,k] = 1 for all valid i,j,k? No.
    
    Let me reconsider. The matrix multiplication tensor T is defined such that:
    T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].
    
    More precisely, the tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    A rank-1 decomposition term (U_c, V_c, W_c) contributes:
    U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i] to the (i,k) output.
    
    For the schoolbook method, we have n*m*p terms, one for each (i,j,k) triple:
    U_{i*m+j, c} = 1, V_{j*p+k, c} = 1, W_{k*n+i, c} = 1
    
    This gives rank n*m*p = 2*3*6 = 36.
    
    Can we do better? The Strassen algorithm can reduce the rank for square matrices.
    For 2x2 * 2x2, Strassen uses rank 7 instead of 8.
    
    For 2x3 * 3x6, we can potentially use a similar approach. Let's think about this.
    
    Actually, let's try to find a lower rank decomposition. The known lower bounds for
    matrix multiplication tensor rank are complex, but for small dimensions we can try
    to find good decompositions.
    
    One approach: embed the 2x3 * 3x6 multiplication into larger square multiplications
    and use known fast algorithms, then extract the relevant parts.
    
    For example, we can pad A to 3x3 and B to 3x6 or 6x6, use fast matrix multiplication,
    and extract the result. But this might not reduce the rank.
    
    Another approach: use the fact that 2x3 * 3x6 can be seen as a block multiplication.
    
    Let me try a direct search or known decomposition.
    
    Actually, for the purpose of this task, let's start with the schoolbook rank of 36
    and see if we can find a better one. Given the time constraint, let's implement the
    schoolbook first and then try to improve.
    
    The schoolbook decomposition has rank 36. Let's see if there's a known better bound.
    
    For m x n * n x p matrices, the trivial bound is m*n*p. Strassen's algorithm for
    n x n * n x n gives rank 7^n for n=2, so for 2x2*2x2 it's 7 < 8.
    
    For our case, 2x3 * 3x6, we could potentially use a combination of Strassen-like
    decompositions on sub-blocks.
    
    Let me try to construct a decomposition with rank less than 36.
    
    One idea: split the 3x6 matrix B into two 3x3 matrices B1 and B2. Then:
    C = A * B = A * [B1, B2] = [A*B1, A*B2]
    
    Each A*B1 and A*B2 is 2x3 * 3x3. If we can decompose 2x3*3x3 with rank r,
    then the total rank would be 2*r.
    
    For 2x3 * 3x3, the schoolbook rank is 2*3*3 = 18. Can we do better?
    
    Similarly, we could split A into two 1x3 rows, and each 1x3 * 3x6 is a dot product
    with each column of B, which is 6 scalar multiplications per row, so 12 total.
    That's the same as schoolbook.
    
    Let me think differently. The tensor for 2x3 * 3x6 multiplication has dimension
    (2,3,6). The rank of this tensor is what we're looking for.
    
    Known results: The rank of the matrix multiplication tensor <n,m,p> is at most
    n*m*p (schoolbook). For <2,2,2>, the rank is 7 (Strassen).
    
    For <2,3,6>, I'm not sure of the exact minimum rank, but let's try to find a
    decomposition with rank less than 36.
    
    Given the time constraint, let's implement the schoolbook decomposition first
    and ensure it's correct, then try to improve if time permits.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Schoolbook decomposition: rank = n * m * p
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
    
    result = {"U": U, "V": V, "W": W}
    
    # Check if we have time to try to improve
    if time.time() < deadline:
        # Try to find a better decomposition
        # For now, return the schoolbook result
        pass
    
    return result
# EVOLVE-BLOCK-END
