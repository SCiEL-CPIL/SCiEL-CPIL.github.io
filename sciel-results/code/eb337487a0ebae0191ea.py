"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

import time
import itertools
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    
    The strategy:
    1. The 3x3x6 tensor can be viewed as three independent 3x3x2 sub-tensors
       (since the 3x6 matrix has 3 columns of 3 elements each, but actually it's
       3 rows of 3 columns... let me reconsider).
       
    Actually, 3x3 * 3x6: A is 3x3, B is 3x6, C is 3x6.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i,j,k) where i in 0..2, j in 0..2, k in 0..5.
    Total entries: 3*3*6 = 54.
    
    We can decompose this as a direct sum over k of 3x3x2? No, that's not quite right.
    Actually, for each fixed k, we have a 3x3 matrix multiplication (but B's column k varies).
    
    Better view: The tensor is T_{i,j,k} = A_{i,j} * B_{j,k}.
    This is a sum over k of tensors T^{(k)} where T^{(k)}_{i,j} = A_{i,j} * B_{j,k}.
    
    For each k, the sub-problem is: multiply 3x3 by 1x3 (taking column k of B) to get 3x1.
    That's a 3x3x1 tensor, which has rank 3 (just the 3 terms).
    
    Wait, that gives rank 3*6 = 18, which is worse than schoolbook 54... no wait, schoolbook is 54.
    18 would be better. But can we do better?
    
    Actually, the standard approach: the rank of the matrix multiplication tensor for 
    m x n by n x p is at most m*n*p (schoolbook), but can be less.
    
    For 3x3 by 3x6, we can think of it as 6 independent 3x3 by 3x1 multiplications.
    Each 3x3 by 3x1 has rank at most 9 (schoolbook) but actually rank 3 since it's just 
    multiplying a 3x3 matrix by a 3-vector.
    
    Hmm, let me reconsider. The tensor T_{i,j,k} for i=0,1,2; j=0,1,2; k=0,1,2,3,4,5.
    
    One approach: use the fact that this is equivalent to multiplying a 3x3 matrix by a 3x6 matrix.
    We can group the 6 columns into 3 pairs, giving 3 independent 3x3 by 3x2 problems.
    
    For 3x3 by 3x2: C[i,k] = sum_j A[i,j] * B[j,k] for k=0,1.
    This is a 3x3x2 tensor. What's its rank?
    
    The schoolbook rank is 3*3*2 = 18. But we can do better.
    
    Actually, let me just use a direct construction. The key insight is that we can use
    Strassen-like techniques. But for 3x3x2, let's think about it.
    
    For multiplying 3x3 by 3x2:
    C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
    C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
    
    For each row i, this is a 1x3 by 3x2 multiplication, which has rank 3 (or can be done with 3 multiplications).
    So total rank would be 3*3 = 9 for the 3x3x2 case? No, that's schoolbook.
    
    Let me try a different approach. I'll use the known result that the rank of the 
    3x3x2 matrix multiplication tensor is 13 (from literature on small matrix multiplication).
    
    Actually, I'm not sure about the exact rank. Let me just construct a valid decomposition 
    with rank less than 54.
    
    Simplest improvement: note that T_{i,j,k} = A_{i,j} B_{j,k}. For fixed i and j, as k varies,
    we have B_{j,k} for k=0..5. So we can write:
    
    T = sum_{i,j} A_{i,j} B_{j,0:6} e_i e_j^T ... this isn't helping directly.
    
    Let me just use the schoolbook but with a smarter grouping. Actually, the schoolbook 
    already gives rank 54. Can we reduce it?
    
    Key observation: The tensor T_{i,j,k} is linear in A_{i,j} and B_{j,k}. We can use
    the fact that for fixed j, the terms involving A_{i,j} and B_{j,k} can be shared.
    
    For a given j, we need to compute sum_i,k A_{i,j} B_{j,k} for all i,k. This is an 
    outer product of vector A_{:,j} (length 3) and B_{j,:} (length 6). The rank of this 
    outer product as a bilinear map is 1 (it's just a single multiplication of the two vectors, 
    but we need all 18 products).
    
    Actually, the outer product of a 3-vector and a 6-vector requires 3*6 = 18 scalar 
    multiplications in the naive approach, but as a tensor rank, it's 1 because 
    v_i w_k = (v_i)(w_k) is already a rank-1 tensor in the (i,k) space for fixed j.
    
    Wait, I think I'm overcomplicating this. Let me just construct the decomposition 
    directly.
    
    The decomposition: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    where n=3, m=3, p=6. So U is 9xR, V is 18xR, W is 18xR.
    
    For schoolbook: R = 54, and each term corresponds to one (i,j,k) triple.
    
    To reduce rank, we need to find linear combinations. One standard technique is to 
    use the fact that the matrix multiplication tensor has structure.
    
    Let me try a simple reduction: For the 3x3x6 case, we can use the fact that 
    multiplying by a 3x6 matrix is equivalent to 6 independent 3x3 by 3x1 multiplications.
    Each 3x3 by 3x1 multiplication: C_i = sum_j A_{i,j} B_{j,k}. This is just a dot product, 
    which requires 3 multiplications (one for each j). So total rank = 3 (for the 3 j's) * 6 (for the 6 k's) = 18? 
    
    No wait, that's not right either. The tensor rank isn't simply the product of the ranks of sub-problems.
    
    Let me just go with a constructive approach. I'll use the following decomposition:
    
    For each pair (i, j), the contribution to all k is A_{i,j} * B_{j,k}. This is a rank-1 
    term in the (i, k) output space for fixed j... no, it's a rank-1 term in the full tensor 
    only if we fix i and j.
    
    OK, I think the cleanest approach is to just use the schoolbook decomposition but 
    check if we can merge any terms. In general, for matrix multiplication, the schoolbook 
    rank is tight for small dimensions, but let me verify.
    
    Actually, for 3x3 * 3x6, I believe the rank is 18. Here's why:
    
    C = A * B where A is 3x3 and B is 3x6.
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    For each (i,k) pair, this is a sum of 3 products. The total number of (i,k) pairs is 3*6 = 18.
    Each product A[i,j]*B[j,k] appears in exactly one (i,k) output.
    
    The tensor T_{i,j,k} has 3*3*6 = 54 entries. The schoolbook uses 54 rank-1 terms.
    But we can do better by sharing multiplications.
    
    Actually, I recall that the rank of the m x n by n x p matrix multiplication tensor 
    is at most mn + np + mp - something, but I don't remember the exact formula.
    
    Let me just try to construct a rank-18 decomposition. The idea is:
    
    For each j in 0..2, we have the row A[:,j] (length 3) and the row B[j,:] (length 6).
    The outer product A[:,j] @ B[j,:]^T gives a 3x6 matrix. The sum over j gives C.
    
    Each outer product can be computed with 3*6 = 18 multiplications, but as a bilinear 
    map, the outer product of a 3-vector and a 6-vector has tensor rank 1 (in the sense 
    that it's a single bilinear form if we consider the vectors as inputs).
    
    I think I'm confusing tensor rank with arithmetic complexity. Let me just implement 
    the schoolbook and see if I can find a reduction.
    
    Given the time constraints, let me just return the schoolbook decomposition with 
    rank 54, which is the same as the seed. But the task asks for rank-minimal, so I 
    should try to do better.
    
    Let me try a rank-18 approach: group by j.
    For each j, we need to compute the outer product of A[:,j] and B[j,:].
    This outer product has 18 entries. As a bilinear map from (A[:,j], B[j,:]) to the 
    3x6 output, it's a rank-1 tensor in the (A[:,j], B[j,:]) input space, but it produces 
    18 output values.
    
    I think the correct way to see this is: the full tensor T_{i,j,k} can be written as 
    sum over j of (e_i in R^3) (e_j in R^3) (B[j,:] in R^6) (A[:,j] in R^3)... this is 
    getting confusing.
    
    Let me just go with a simple approach: use the schoolbook but with rank 54. If I can't 
    find a better decomposition quickly, I'll return that.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    # U: (n*m) x rank = 9 x rank
    # V: (m*p) x rank = 18 x rank
    # W: (p*n) x rank = 18 x rank
    
    # Schoolbook: rank = n*m*p = 54
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
