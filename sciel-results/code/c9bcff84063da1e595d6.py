"""Rewritten program for 2x3x7 matrix multiplication tensor rank minimization."""
import time
import itertools

DIMENSIONS = [2, 3, 7]

def construct():
    """
    Construct a low-rank exact decomposition of the 2x3x7 matrix multiplication tensor.
    
    Strategy:
    - The schoolbook rank is 2*3*7 = 42.
    - We can do better by exploiting structure.
    - Key insight: For multiplying a 2x3 matrix by a 3x7 matrix, we can use 
      the fact that the inner dimension is 3. 
    - We can decompose this using a combination of Strassen-like techniques on 
      the 2x2 blocks and efficient handling of the 7-length dimension.
    
    A proven approach: Use the fact that we can treat the 3x3 inner product 
    with rank-5 algorithms (since 3x3 matrix multiplication has rank 5 via 
    specific decompositions, though standard is 27, Strassen gives 7 for 2x2).
    
    Actually, let's think more carefully. The tensor is T[i,j,k] where 
    i in {0,1} (rows of A), j in {0,1,2} (cols of A / rows of B), k in {0..6} (cols of B).
    The multiplication is C[i,k] = sum_j A[i,j] * B[j,k].
    
    We want to write this as sum_{c=1}^{rank} U[i,j,c] * V[j,k,c] * W[k,i,c]
    where the tensor element is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    Here n=2, m=3, p=7.
    
    One known result: The rank of matrix multiplication tensor M_{n,m,p} is at most 
    n*m*p, but can be lower. For M_{2,3,7}, we can try to find a decomposition.
    
    Let's use a constructive approach. We can split the 3 columns of A (and 3 rows of B)
    into parts. 
    
    Alternative: Use the fact that we can compute products more efficiently by 
    combining terms. 
    
    Let me try a direct construction with rank less than 42.
    
    Actually, for the specific case of 2x3x7, a known lower bound is not trivially 
    less than 42, but we can definitely do better than 42 using standard tricks.
    
    Let's use a hybrid approach:
    - For the 2x3 part, we can use a rank-6 decomposition (since 2x3 matrix-vector 
      product can be done with 6 multiplications, but here it's matrix-matrix).
    
    Let me think differently. The tensor T has indices (i,j,k) with i in [2], j in [3], k in [7].
    T[i,j,k] = 1 for all valid indices (it's the identity tensor for multiplication).
    
    We want to decompose T as a sum of rank-1 tensors:
    T[i,j,k] = sum_c U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    This is equivalent to finding matrices U_c, V_c, W_c such that for each c,
    the rank-1 tensor U_c ⊗ V_c ⊗ W_c sums to T.
    
    A simple improvement: Notice that we can group the j dimension. For each fixed j,
    we have a 2x7 sub-tensor. But that doesn't immediately help.
    
    Let's try a different strategy: Use the fact that we can compute the product 
    by first computing some intermediate combinations.
    
    Actually, let's use a known decomposition. For M_{2,3,7}, one can use the 
    following idea: Split the 3 rows of B (and 3 cols of A) into two parts: 
    one row and two rows. Then use Strassen on the 2x2 part and schoolbook on the rest.
    
    But a cleaner approach: Let's just implement a rank-39 decomposition based on 
    the observed improvement from island 9. The knowledge says a hybrid Strassen/Schoolbook 
    achieved rank 39.
    
    Let me construct that:
    - Split the inner dimension j={0,1,2} into j0={0} and j1={1,2}.
    - For j1 (2 rows), use Strassen's algorithm on the 2x2 block (rows i={0,1}, cols j={1,2} for A, 
      and rows j={1,2}, cols k for B). This gives 7 multiplications instead of 2*2=4... 
      wait, Strassen for 2x2x2 gives 7 instead of 8.
    
    Actually, the 2x2x2 matrix multiplication tensor has rank 7 (Strassen). 
    So for the sub-problem of multiplying a 2x2 matrix by a 2x7 matrix, we can 
    use Strassen on the 2x2 part, giving 7*7=49? No, that's not right.
    
    Let me reconsider. The full problem is 2x3 times 3x7. 
    We can view it as: C = A * B where A is 2x3, B is 3x7.
    
    Split A into A_left (2x1) and A_right (2x2), and B into B_top (1x7) and B_bottom (2x7).
    Then C = A_left * B_top + A_right * B_bottom.
    
    For A_left * B_top: this is (2x1)(1x7) = 2x7, which requires 2*1*7 = 14 multiplications 
    in schoolbook, but actually it's just 2*7=14 scalar products, each being a single multiplication. 
    So rank contribution is 14.
    
    For A_right * B_bottom: this is (2x2)(2x7) = 2x7. Using Strassen on the 2x2 part, 
    we need 7 "multiplications" where each "multiplication" is actually a product of a linear 
    combination of A entries with a linear combination of B entries, producing a 7-vector. 
    So each Strassen multiplication produces a 7-element vector, meaning we need 7*7=49? 
    No, that's not how it works.
    
    In Strassen's algorithm for 2x2x2, the 7 products are of the form:
    m1 = (a11+a22)(b11+b22)
    m2 = (a21+a22)b11
    m3 = a11(b12-b22)
    m4 = a22(b21-b11)
    m5 = (a11+a12)b22
    m6 = (a21-a11)(b11+b12)
    m7 = (a12-a22)(b21+b22)
    
    Each of these is a scalar product. When we extend to 2x2 times 2x7, each "scalar product" 
    becomes a product of a scalar (linear combo of A entries) with a 7-vector (linear combo of 
    B rows), giving a 7-vector result. So we have 7 such products, each contributing to all 7 
    columns. The rank contribution is 7 (one per Strassen product), but each rank-1 term 
    involves a 7-vector in the k dimension.
    
    Wait, I need to be more careful. In the tensor decomposition, each rank-1 term is 
    U[i,j] * V[j,k] * W[k,i]. For the Strassen part, each of the 7 products corresponds to 
    one rank-1 term in the tensor, but V[j,k] is a 3x7 matrix (for the full problem) or 
    2x7 for the sub-problem.
    
    This is getting complicated. Let me just use the known rank-39 construction.
    
    Based on the knowledge, the hybrid approach achieved rank 39. Let me implement that.
    
    Actually, let me try a simpler but effective approach: use rank 39 by combining 
    Strassen on a 2x2 block and schoolbook on the remaining part.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # We'll construct a decomposition with rank 39.
    # Strategy: Use Strassen on the first 2 columns of A (and first 2 rows of B), 
    # and schoolbook on the last column/row.
    
    # Strassen for 2x2x2 uses 7 products. For 2x2 times 2x7, each product is a 
    # scalar times a 7-vector, so we get 7 rank-1 terms for the 2x2x7 sub-problem.
    # Schoolbook for 2x1x7 uses 2*1*7 = 14 rank-1 terms.
    # Total: 7 + 14 = 21? That seems too good. Let me verify.
    
    # Actually, for the 2x2x7 part, the tensor has indices (i,j,k) with i in [2], j in [2], k in [7].
    # Strassen decomposes the 2x2x2 tensor into 7 rank-1 terms. When extended to 2x2x7,
    # each of the 7 terms becomes a rank-7 term (since k ranges over 7 values). 
    # So the 2x2x7 tensor has rank at most 7*7 = 49? No, that's not right either.
    
    # Let me think about this more carefully. The 2x2x7 tensor T[i,j,k] = 1 for all i,j,k.
    # We can write this as sum over k of T[i,j,k]. For each fixed k, T[i,j,k] is the 
    # 2x2x1 tensor, which has rank 4 (schoolbook) or can be decomposed using Strassen-like 
    # methods.
    
    # Actually, the rank of the 2x2x7 tensor is the same as the rank of the 2x2 matrix 
    # multiplication tensor tensor-producted with the 7-dimensional identity. 
    # The rank of M_{2,2,1} is 4, and tensoring with I_7 gives rank 4*7 = 28.
    # But Strassen shows M_{2,2,2} has rank 7, so M_{2,2,1} might have rank less than 4? 
    # No, M_{2,2,1} is the standard 2x2 matrix multiplication, which has rank 4 in the 
    # tensor sense (each of the 4 products is a rank-1 term).
    
    # I'm overcomplicating this. Let me just use a known good decomposition.
    
    # Based on the observed result, rank 39 was achieved. Let me construct that.
    
    # Use schoolbook for most terms, but replace some with Strassen-like combinations.
    
    # Simple approach: Start with schoolbook (rank 42) and try to merge terms.
    
    # For now, let's use a rank-39 construction:
    # - Use 7 Strassen products for a 2x2x7 block (treating 2 columns of A, 2 rows of B)
    # - Use schoolbook for the remaining 1 column/row: 2*1*7 = 14 terms
    # - But 7 + 14 = 21, which is less than 39. This suggests my understanding is off.
    
    # Let me just implement a valid decomposition with rank 39 using a known method.
    
    # Actually, the safest approach is to use the schoolbook decomposition (rank 42) 
    # and then try to reduce it. But since we know rank 39 is achievable, let me 
    # implement a specific construction.
    
    # Here's a concrete approach:
    # 1. For the sub-problem of A[:, 0:2] * B[0:2, :] (2x2 times 2x7), use a 
    #    decomposition of rank 14 (instead of 2*2*7=28 schoolbook).
    # 2. For the sub-problem of A[:, 2:3] * B[2:3, :] (2x1 times 1x7), use schoolbook 
    #    with rank 14.
    # 3. Total rank: 14 + 14 = 28? Still seems too low.
    
    # I think the issue is that when we use Strassen, the "multiplications" are not 
    # independent rank-1 terms in the full tensor; they involve linear combinations 
    # that couple different k values.
    
    # Let me just use a conservative rank and implement a valid decomposition.
    
    # For safety, let's use the schoolbook decomposition (rank 42) which is known 
    # to be valid, and then try to optimize.
    
    # Actually, let me implement a rank-39 decomposition based on a known construction.
    
    # The key insight: We can use the fact that for the 3x3 inner product, we can 
    # use a rank-5 decomposition (since 3x3 matrix multiplication can be done with 
    # 5 multiplications in some cases, but standard is 27).
    
    # For now, let's use a simple but valid improvement: rank 39.
    
    # Construction:
    # - Use 7 terms for a Strassen-like decomposition of a 2x2x7 block
    # - Use 32 terms for the rest
    # Total: 39
    
    # Let me just implement the schoolbook for now and return rank 42, since 
    # I'm not confident in the rank-39 construction.
    
    # Actually, the task says the current score is 1.0 (schoolbook, rank 42), and 
    # we know rank 39 is achievable. Let me implement that.
    
    # Here's a concrete rank-39 construction:
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 6 x 39
    V = [[0] * rank for _ in range(m * p)]  # 21 x 39
    W = [[0] * rank for _ in range(p * n)]  # 14 x 39
    
    # We'll use a hybrid approach:
    # - For j=0,1 (first two columns of A), use a Strassen-like decomposition
    # - For j=2 (last column of A), use schoolbook
    
    # Strassen-like for 2x2x7:
    # The 7 products in Strassen for 2x2x2, extended to 2x2x7:
    # Each product m_c = (linear combo of A entries) * (linear combo of B entries)
    # where the B entry is a 7-vector.
    
    # Let me define the 7 Strassen terms for the 2x2 part (rows i=0,1; cols j=0,1):
    # m1 = (A[0,0]+A[1,1]) * (B[0,:]+B[1,:])
    # m2 = (A[1,0]+A[1,1]) * B[0,:]
    # m3 = A[0,0] * (B[1,:]-B[0,:])
    # m4 = A[1,1] * (B[1,:]-B[0,:])  # Wait, this is not quite right
    # Let me use the standard Strassen formulas:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # For our 2x2x7 case, a_ij = A[i,j] for i in {0,1}, j in {0,1}
    # b_jk = B[j,k] for j in {0,1}, k in {0..6}
    
    # Each m_c is a 7-vector (since it's a scalar times a 7-vector, or sum of such).
    
    # In the tensor decomposition, each m_c corresponds to one rank-1 term where:
    # U_c[i,j] is the coefficient of A[i,j] in the scalar part
    # V_c[j,k] is the coefficient of B[j,k] in the vector part
    # W_c[k,i] is the coefficient in the output
    
    # This is complex. Let me just use a simpler valid decomposition.
    
    # For now, let's use the schoolbook decomposition with rank 42, which is known 
    # to be valid. This gives score 1.0, which is the baseline.
    
    # To improve, I need a valid rank < 42 decomposition. Let me use rank 39.
    
    # Here's a simpler approach: Use schoolbook but merge some terms.
    
    # Actually, let me just return the schoolbook decomposition for now, since 
    # I'm not confident in the rank-39 construction details.
    
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
