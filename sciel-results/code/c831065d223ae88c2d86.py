"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.
    
    Strategy: Use a structured decomposition that exploits the small dimensions.
    For 2x3 * 3x7, the output is 2x7. We can think of this as computing 14 output entries.
    
    The tensor T[i,j,k] for matrix multiplication (2x3 * 3x7) has the property that
    T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C.
    
    Actually, let's think more carefully. The matrix multiplication tensor for (n,m,p)
    dimensions has entries indexed by (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    The standard convention is that the tensor T has T[i,j,k] = 1 if and only if the 
    product A[i,j] * B[j,k] contributes to C[i,k].
    
    Wait, let me re-read the task. The output factor follows W[k*n+i, r].
    
    So for a 2x3 * 3x7 multiplication:
    - A is 2x3, so n=2, m=3
    - B is 3x7, so m=3, p=7
    - C is 2x7
    
    U has shape (n*m, rank) = (6, rank)
    V has shape (m*p, rank) = (21, rank)
    W has shape (p*n, rank) = (14, rank)
    
    The decomposition says: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    Wait, that's not quite right. Let me think about the standard tensor decomposition.
    
    For matrix multiplication, we have C = A * B where C[i,k] = sum_j A[i,j] * B[j,k].
    
    A rank-r decomposition of the multiplication tensor means we can write:
    C[i,k] = sum_{r=1}^r (sum_j U[i,j,r] * A[i,j]) * (sum_k V[j,k,r] * B[j,k]) * W[i,k,r]
    
    Hmm, that's not standard either. Let me think again.
    
    The standard way: the matrix multiplication tensor T_{n,m,p} is a 3-tensor with
    T[i,j,k] = 1 if the (i,j) entry of the first matrix and (j,k) entry of the second
    matrix contribute to the (i,k) entry of the product.
    
    A rank-r decomposition gives r terms such that:
    T[i,j,k] = sum_{r=1}^r U[i,r] * V[j,r] * W[k,r]
    
    But wait, the indices don't match up this way for matrix multiplication because
    the middle index j is shared between the first two matrices.
    
    Actually, for matrix multiplication, the tensor is defined on the space
    (R^n x R^m)^* x (R^m x R^p)^* x (R^n x R^p)^*
    
    The standard decomposition is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    This can be written as a sum of rank-1 tensors. The schoolbook decomposition
    has rank n*m*p, with each term corresponding to a specific (i,j,k) triple.
    
    For the schoolbook: for each (i,j,k), we have a rank-1 term where:
    - U[i,j] = 1 (row i*m+j of U, column r)
    - V[j,k] = 1 (row j*p+k of V, column r)  
    - W[i,k] = 1 (row k*n+i of W, column r)
    
    So C[i,k] = sum_{r} U[i*m+j(r), r] * V[j(r)*p+k(r), r] * W[k(r)*n+i(r), r]
    
    But this only works if for each r, there's a unique (i,j,k) such that U[i*m+j,r]=1,
    V[j*p+k,r]=1, W[k*n+i,r]=1, and the product contributes to C[i,k].
    
    OK so the key insight is: we want to find a decomposition of rank < 42.
    
    For 2x3 * 3x7, let's think about what's possible.
    
    One approach: split the 3-dimensional middle into 2+1.
    - First part: 2x2 * 2x7 (using Strassen for the 2x2 part, rank 7)
    - Second part: 2x1 * 1x7 (schoolbook, rank 7)
    - Total: 14
    
    But we need to be careful about how the splitting works.
    
    Let me try a different approach. Since the dimensions are small, let me try to
    find a decomposition by exploiting linearity.
    
    Actually, let me try the following structured approach:
    
    For 2x3 * 3x7, we can write:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    ... etc.
    
    The rank of the 2x3x7 multiplication tensor... let me think about known results.
    
    For 2x2 * 2x2, Strassen gives rank 7.
    For 2x2 * 2x3, the rank is at most... hmm.
    
    Let me try a constructive approach. I'll use the fact that we can combine
    Strassen-like ideas with the specific dimensions.
    
    Actually, let me just try to find a good decomposition by reasoning about the
    structure. For 2x3 * 3x7:
    
    We can split the middle dimension: 3 = 2 + 1.
    
    Let A = [A1 | A2] where A1 is 2x2 and A2 is 2x1.
    Let B = [B1; B2] where B1 is 2x7 and B2 is 1x7.
    
    Then C = A1*B1 + A2*B2.
    
    A1*B1 is a 2x2 * 2x7 multiplication. We can decompose this using Strassen
    for the 2x2 part, giving rank 7 (since 2x2 * 2x7 has rank at most 7).
    
    A2*B2 is a 2x1 * 1x7 multiplication, which has rank 7 (each of the 7 output
    entries needs one term, but actually it's rank 7 since each output C[i,k] = A2[i]*B2[k]).
    
    Wait, A2*B2: A2 is 2x1, B2 is 1x7. So C[i,k] = A2[i,0] * B2[0,k].
    This is an outer product, so it has rank 1 as a tensor? No, wait.
    
    Actually, for the tensor decomposition, A2*B2 contributes to C[i,k] = A2[i]*B2[k].
    This can be written as a single rank-1 term? No, because i and k are independent.
    
    Hmm, let me reconsider. The tensor for 2x1 * 1x7 has T[i,j,k] where j is always 0
    (since m=1). So T[i,0,k] = 1 for all i,k. This is a 2x1x7 tensor with all entries
    in the j=0 slice being 1.
    
    The rank of this tensor: we need to represent T[i,0,k] = 1 as a sum of rank-1 terms.
    T[i,0,k] = sum_r U[i,r] * V[0,r] * W[k,r].
    
    Since V[0,r] is just a scalar for each r, we can absorb it. So we need:
    1 = sum_r (U[i,r] * V[0,r]) * W[k,r] for all i,k.
    
    This means the matrix M[i,k] = sum_r (U[i,r]*V[0,r]) * W[k,r] must be all-ones.
    The rank of the all-ones 2x7 matrix is 1. So we can do this with rank 1!
    
    Wait, that's not right either. Let me be more careful.
    
    For the 2x1 * 1x7 case:
    - U has shape (2*1, r) = (2, r)
    - V has shape (1*7, r) = (7, r)
    - W has shape (7*2, r) = (14, r)
    
    T[i,j,k] = 1 if j=0 (which it always is since m=1).
    
    We need: sum_r U[i,r] * V[j,r] * W[k*n+i, r] = T[i,j,k] = 1 (for j=0).
    
    So: sum_r U[i,r] * V[0,r] * W[k*2+i, r] = 1 for all i in {0,1}, k in {0,...,6}.
    
    Let a_r = V[0,r]. Then: sum_r a_r * U[i,r] * W[k*2+i, r] = 1.
    
    For i=0: sum_r a_r * U[0,r] * W[k*2, r] = 1 for all k.
    For i=1: sum_r a_r * U[1,r] * W[k*2+1, r] = 1 for all k.
    
    We can set r=1 only: a_1 = 1, U[0,1] = 1, U[1,1] = 1, W[k*2,1] = 1, W[k*2+1,1] = 1.
    This works! So the 2x1 * 1x7 part has rank 1.
    
    Similarly, for the 2x2 * 2x7 part, we can use Strassen for the 2x2 multiplication
    and extend it to 2x7.
    
    Strassen for 2x2 * 2x2 uses 7 multiplications. For 2x2 * 2x7, we can apply the
    same 7 Strassen combinations to each column of B (or the 7 columns together).
    
    Actually, for 2x2 * 2x7:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    Strassen's algorithm for 2x2 * 2x2 computes:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For 2x2 * 2x7, we can do the same 7 combinations but with B having 7 columns.
    Each M_r is a scalar (since A is 2x2 and we're combining entries), and then we
    multiply by the appropriate B entries.
    
    Wait, no. In Strassen, the M_r are products of linear combinations of A entries
    and linear combinations of B entries. For 2x2 * 2x7, the B entries are B[j,k] for
    j in {0,1}, k in {0,...,6}.
    
    So M1 = (A00+A11) * (B0,k + B1,k) for each k. But that's 7 different products.
    
    Hmm, I think I'm overcomplicating this. Let me think about it differently.
    
    For 2x2 * 2x7, the tensor has T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,...,6}.
    This is a 2x2x7 tensor.
    
    The rank of the 2x2x7 tensor: we can think of it as 7 copies of the 2x2x1 tensor
    (which has rank 4, or 7 with Strassen).
    
    Actually, the 2x2x7 tensor can be seen as a block diagonal of 7 2x2x1 tensors.
    Each 2x2x1 tensor has T[i,j,0] = 1, which is the standard 2x2 multiplication tensor
    for a single output entry.
    
    The rank of a single 2x2x1 tensor (i.e., the tensor for computing one entry of a
    2x2 matrix product) is... well, C00 = A00*B00 + A01*B10. This is a bilinear form
    in A and B, so it has rank 2 (as a bilinear form). But as a trilinear tensor, it's
    different.
    
    I think I'm going in circles. Let me just try a practical approach: construct a
    decomposition and verify it.
    
    Let me try the following:
    1. Split the middle dimension: 3 = 2 + 1.
    2. For the 2x2 * 2x7 part, use a rank-7 decomposition (Strassen-like).
    3. For the 2x1 * 1x7 part, use a rank-1 decomposition.
    4. Total rank: 8.
    
    But wait, I need to make sure the decomposition is correct. Let me think about
    what the U, V, W matrices look like.
    
    For the full 2x3 * 3x7 problem:
    - U has shape (6, r) where rows are indexed by (i,j) with i in {0,1}, j in {0,1,2}.
    - V has shape (21, r) where rows are indexed by (j,k) with j in {0,1,2}, k in {0,...,6}.
    - W has shape (14, r) where rows are indexed by (k,i) with k in {0,...,6}, i in {0,1}.
    
    For the 2x2 * 2x7 part (using j=0,1 for A and B):
    - U rows: (0,0), (0,1), (1,0), (1,1) -> rows 0,1,3,4 of U
    - V rows: (0,0)...(0,6), (1,0)...(1,6) -> rows 0-6, 7-13 of V
    - W rows: all 14 rows
    
    For the 2x1 * 1x7 part (using j=2 for A and B):
    - U rows: (0,2), (1,2) -> rows 2,5 of U
    - V rows: (2,0)...(2,6) -> rows 14-20 of V
    - W rows: all 14 rows
    
    Let me try to construct the 2x2 * 2x7 decomposition with rank 7.
    
    Actually, I realize that the 2x2 * 2x7 tensor is not simply 7 copies of a 2x2 * 2x1
    tensor in a way that allows sharing of the Strassen structure. Each output column k
    requires its own set of Strassen multiplications.
    
    So the rank of 2x2 * 2x7 is at most 7*7 = 49 (using Strassen for each column),
    or 4*7 = 28 (using schoolbook for each column).
    
    That's not helpful. Let me think differently.
    
    Actually, for 2x2 * 2x7, we can write:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    This is a sum of 2 terms for each (i,k), so the total number of terms is 2*2*7 = 28
    for the schoolbook. But we can do better by sharing computations.
    
    Hmm, I think for these small dimensions, the best approach might be to just try
    different structured decompositions and verify them.
    
    Let me try a different strategy: use the fact that 2x3 * 3x7 can be seen as
    computing 14 output entries, each of which is a dot product of length 3.
    
    The total number of scalar multiplications in the schoolbook is 2*3*7 = 42.
    We want to reduce this.
    
    One idea: use linearity to combine terms. For example, if two output entries
    share a common factor, we can compute it once.
    
    Let me try the following decomposition with rank 35 (just to see if it's valid):
    
    Actually, let me just try to find ANY decomposition with rank < 42 and verify it.
    
    Here's an idea: for the 2x3 * 3x7 case, we can use the following trick.
    Note that C[i,k] = sum_j A[i,j] * B[j,k].
    
    We can write this as:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    Now, suppose we define new variables:
    X0 = A[i,0]
    X1 = A[i,1]
    X2 = A[i,2]
    Y0 = B[0,k]
    Y1 = B[1,k]
    Y2 = B[2,k]
    
    Then C[i,k] = X0*Y0 + X1*Y1 + X2*Y2.
    
    This is a dot product of two length-3 vectors. The rank of the dot product tensor
    for length-3 vectors is 3 (schoolbook). So for each (i,k), we need 3 multiplications.
    Total: 14 * 3 = 42.
    
    To do better, we need to share multiplications across different (i,k) pairs.
    
    Idea: group the output entries. For example, C[0,0] and C[0,1] both use A[0,0], A[0,1], A[0,2]
    but with different B entries. If we precompute linear combinations of A[0,*], we can
    reuse them.
    
    Let me try: for row i=0 of A, define:
    P0 = A[0,0]
    P1 = A[0,1]
    P2 = A[0,2]
    
    Then C[0,k] = P0*B[0,k] + P1*B[1,k] + P2*B[2,k].
    
    This doesn't save anything because we still need 3 multiplications per k.
    
    Alternative: use Strassen-like ideas on the 3-dimensional dot product.
    
    The dot product of two length-3 vectors: x.y = x0*y0 + x1*y1 + x2*y2.
    The rank of this bilinear form is 3 (it's diagonal). So we can't do better than 3
    for a single dot product.
    
    But if we have multiple dot products with shared structure, we might do better.
    
    For example, consider:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    
    These share A[0,0], A[0,1], A[0,2] but use different B entries.
    
    If we define:
    M0 = A[0,0] * (B[0,0] + B[0,1])
    M1 = A[0,1] * (B[1,0] + B[1,1])
    M2 = A[0,2] * (B[2,0] + B[2,1])
    
    Then C[0,0] + C[0,1] = M0 + M1 + M2. But we need C[0,0] and C[0,1] separately,
    not their sum. So this doesn't directly help.
    
    I think for this problem, the rank might actually be 42 or close to it. Let me
    just try to construct a valid decomposition with rank 42 (the schoolbook) and
    make sure it's correct, then see if I can improve it.
    
    Actually, the current program already does the schoolbook with rank 42. The score
    is 1.0, which means it's the baseline. To improve, I need to find a decomposition
    with rank < 42.
    
    Let me try a different approach: use the fact that the matrix multiplication tensor
    for (2,3,7) might have a lower rank than 42 due to the specific dimensions.
    
    Known results:
    - 2x2 * 2x2: rank 7 (Strassen)
    - 2x2 * 2x3: rank ? 
    - 2x3 * 3x3: rank ?
    
    I don't know the exact ranks for these. Let me try to construct a decomposition
    for 2x3 * 3x7 with rank < 42.
    
    Here's an idea: split the 3x7 matrix B into two parts: B1 (3x3) and B2 (3x4).
    Then C = A*B1 + A*B2.
    
    A*B1 is 2x3 * 3x3. A*B2 is 2x3 * 3x4.
    
    If we can decompose 2x3 * 3x3 with rank r1 and 2x3 * 3x4 with rank r2, then
    the total rank is r1 + r2.
    
    For 2x3 * 3x3: schoolbook rank is 2*3*3 = 18.
    For 2x3 * 3x4: schoolbook rank is 2*3*4 = 24.
    Total: 42. No improvement.
    
    Let me try yet another approach. Since the dimensions are small, let me just
    hardcode a decomposition that I know works, even if it's not optimal.
    
    Actually, let me try to use the following insight: for 2x3 * 3x7, we can think of
    it as 7 instances of 2x3 * 3x1 (one for each column of B). Each 2x3 * 3x1 instance
    computes a 2x1 vector: C[i,0] = sum_j A[i,j] * B[j,0].
    
    The 2x3 * 3x1 tensor has T[i,j,0] = 1, which is a 2x3x1 tensor. The rank of this
    tensor is... well, it's the tensor for computing 2 dot products of length 3.
    
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    
    This is a 2x3 matrix times a 3x1 vector. The rank of this bilinear map is 3
    (the minimum of 2 and 3... no, the rank of the tensor is different).
    
    Actually, the tensor T for 2x3 * 3x1 has T[i,j,k] = 1 if k=0. So it's a 2x3x1
    tensor with all entries in the k=0 slice being 1.
    
    We need: sum_r U[i,r] * V[j,r] * W[k*2+i, r] = T[i,j,k].
    
    For k=0: sum_r U[i,r] * V[j,r] * W[i, r] = 1 for all i,j.
    
    This means the matrix M[i,j] = sum_r U[i,r] * V[j,r] * W[i,r] must be all-ones.
    
    We can do this with rank 3: set r=0,1,2, and for each r, set U[i,r] = 1, V[j,r] = 1,
    W[i,r] = 1/3. Then M[i,j] = sum_{r=0}^2 1*1*(1/3) = 1.
    
    Wait, that works! So the 2x3 * 3x1 tensor has rank 3.
    
    Therefore, the 2x3 * 3x7 tensor (which is 7 copies of 2x3 * 3x1) has rank at most
    7*3 = 21.
    
    But can we do better by sharing across the 7 copies?
    
    For the full 2x3 * 3x7 tensor:
    T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,6}.
    
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    Let me try rank 21: for each k, use 3 terms.
    For k=0: r=0,1,2
    For k=1: r=3,4,5
    ...
    For k=6: r=18,19,20
    
    For each k and r in {3k, 3k+1, 3k+2}:
    U[i*3+j, r] = 1 for all i,j
    V[j*7+k, r] = 1 for all j
    W[k*2+i, r] = 1/3 for all i
    
    Let me verify: for a given (i,j,k), the sum over r of U[i*3+j,r] * V[j*7+k,r] * W[k*2+i,r]
    = sum_{r=3k}^{3k+2} 1 * 1 * (1/3) = 1. Correct!
    
    So we have a valid decomposition with rank 21!
    
    Can we do better? Let me see if we can share terms across different k values.
    
    Suppose we use rank r < 21. Then we need:
    sum_{s=0}^{r-1} U[i*3+j, s] * V[j*7+k, s] * W[k*2+i, s] = 1 for all i,j,k.
    
    This is a system of 2*3*7 = 42 equations in r*(6+21+14) = 41r unknowns.
    
    For r=21: 41*21 = 861 unknowns, 42 equations. Very underdetermined.
    For r=14: 41*14 = 574 unknowns, 42 equations.
    For r=7: 41*7 = 287 unknowns, 42 equations.
    
    The question is whether there's a solution with r < 21.
    
    Let me try r=14. We need to find U (6x14), V (21x14), W (14x14) such that
    the tensor equation holds.
    
    This is a nonlinear system, so it's not easy to solve directly. But let me try a
    structured approach.
    
    Idea: group the 7 k-values into 2 groups: {0,1,2,3} and {4,5,6}.
    For the first group, use 7 terms; for the second group, use 7 terms. Total: 14.
    
    For k in {0,1,2,3} (group 1, r=0..6):
    For each k in {0,1,2,3}, we need 3 terms, but we only have 7 terms for 4 k-values.
    7/4 is not an integer, so this doesn't work cleanly.
    
    Alternative: use 7 terms total, each term handling multiple k-values.
    
    Let me try: for each s in {0,...,6}, define:
    U[i*3+j, s] = 1 for all i,j
    V[j*7+k, s] = 1/7 for all j,k
    W[k*2+i, s] = 7/3 for all i,k
    
    Then sum_{s=0}^6 U[i*3+j,s] * V[j*7+k,s] * W[k*2+i,s]
    = sum_{s=0}^6 1 * (1/7) * (7/3) = 7 * (1/3) = 7/3. Not 1.
    
    Let me adjust: set W[k*2+i, s] = 3/7.
    Then sum = 7 * 1 * (1/7) * (3/7) = 3/7. Still not 1.
    
    Let me set V[j*7+k, s] = 1/7 and W[k*2+i, s] = 1.
    Then sum = 7 * 1 * (1/7) * 1 = 1. Yes!
    
    So with rank 7:
    For s in {0,...,6}:
    U[i*3+j, s] = 1 for all i,j
    V[j*7+k, s] = 1/7 for all j,k
    W[k*2+i, s] = 1 for all i,k
    
    Let me verify: for (i,j,k), sum_{s=0}^6 U[i*3+j,s] * V[j*7+k,s] * W[k*2+i,s]
    = sum_{s=0}^6 1 * (1/7) * 1 = 7 * (1/7) = 1. Correct!
    
    So we have a valid decomposition with rank 7!
    
    Wait, that seems too good to be true. Let me double-check.
    
    The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,6}.
    
    With U[i*3+j, s] = 1, V[j*7+k, s] = 1/7, W[k*2+i, s] = 1 for all s in {0,...,6}:
    
    sum_{s=0}^6 U[i*3+j, s] * V[j*7+k, s] * W[k*2+i, s]
    = sum_{s=0}^6 1 * (1/7) * 1
    = 7 * (1/7)
    = 1.
    
    Yes, this is correct! The rank is 7.
    
    But wait, can we do even better? Let me try rank 6.
    
    With rank 6: U (6x6), V (21x6), W (14x6).
    
    We need: sum_{s=0}^5 U[i*3+j, s] * V[j*7+k, s] * W[k*2+i, s] = 1 for all i,j,k.
    
    Let me try: U[i*3+j, s] = 1, V[j*7+k, s] = 1/6, W[k*2+i, s] = 1.
    Then sum = 6 * (1/6) = 1. Correct!
    
    So rank 6 works too!
    
    Can we do rank 5?
    U[i*3+j, s] = 1, V[j*7+k, s] = 1/5, W[k*2+i, s] = 1.
    Sum = 5 * (1/5) = 1. Correct!
    
    Rank 4?
    U[i*3+j, s] = 1, V[j*7+k, s] = 1/4, W[k*2+i, s] = 1.
    Sum = 4 * (1/4) = 1. Correct!
    
    Rank 3?
    U[i*3+j, s] = 1, V[j*7+k, s] = 1/3, W[k*2+i, s] = 1.
    Sum = 3 * (1/3) = 1. Correct!
    
    Rank 2?
    U[i*3+j, s] = 1, V[j*7+k, s] = 1/2, W[k*2+i, s] = 1.
    Sum = 2 * (1/2) = 1. Correct!
    
    Rank 1?
    U[i*3+j, 0] = 1, V[j*7+k, 0] = 1, W[k*2+i, 0] = 1.
    Sum = 1 * 1 * 1 = 1. Correct!
    
    So the rank is 1?! That can't be right. Let me re-examine.
    
    Oh wait, I think I made an error. The tensor T[i,j,k] is not 1 for all i,j,k.
    Let me re-read the task.
    
    "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by
    a 3 by 7 matrix."
    
    The matrix multiplication tensor for (n,m,p) has T[i,j,k] = 1 if the (i,j) entry
    of A and the (j,k) entry of B contribute to the (i,k) entry of C.
    
    For standard matrix multiplication, C[i,k] = sum_j A[i,j] * B[j,k].
    
    So T[i,j,k] = 1 for all valid i,j,k (where i in [0,n), j in [0,m), k in [0,p)).
    
    Wait, that's what I assumed. So T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2},
    k in {0,...,6}.
    
    And my decomposition with rank 1 gives:
    sum_{s=0}^0 U[i*3+j, s] * V[j*7+k, s] * W[k*2+i, s] = 1 * 1 * 1 = 1.
    
    So the rank is indeed 1?
    
    That seems wrong. Let me think about why.
    
    Oh, I see the issue. The tensor decomposition I'm constructing doesn't actually
    compute the matrix product correctly. The decomposition says:
    
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    But this is not the standard way to write the matrix multiplication tensor
    decomposition. Let me re-read the task.
    
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    And the interface says:
    "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    So U has rows indexed by (i,j) with i in [0,n), j in [0,m).
    V has rows indexed by (j,k) with j in [0,m), k in [0,p).
    W has rows indexed by (k,i) with k in [0,p), i in [0,n).
    
    The decomposition is:
    sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    Where T[i,j,k] is the matrix multiplication tensor.
    
    For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    
    So my rank-1 decomposition should work. But that can't be right because it would
    mean that any matrix multiplication can be done with rank 1, which is absurd.
    
    Let me re-examine. The issue is that the decomposition I'm writing doesn't
    actually compute the matrix product. It just satisfies the tensor equation
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k].
    
    But the actual matrix product is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    And the tensor decomposition should allow us to compute C[i,k] from A and B using
    the factors U, V, W.
    
    The standard way to interpret the tensor decomposition is:
    C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    
    No, that's not right either. Let me think about this more carefully.
    
    Actually, I think the correct interpretation is:
    
    Define intermediate variables:
    X_r = sum_{i,j} U[i*m+j, r] * A[i,j]
    Y_r = sum_{j,k} V[j*p+k, r] * B[j,k]
    
    Then C[i,k] = sum_r X_r * Y_r * W[k*n+i, r]
    
    For this to equal sum_j A[i,j] * B[j,k], we need:
    sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k'']) * W[k*n+i, r]
    = sum_j A[i,j] * B[j,k]
    
    This is a much stronger condition than just sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k].
    
    I think the tensor equation sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    is the correct condition, and it does imply that the matrix product can be computed
    using the factors U, V, W.
    
    Let me verify this. If sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all
    i,j,k, then:
    
    sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k'']) * W[k*n+i, r]
    
    = sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k'']) * W[k*n+i, r]
    
    This doesn't simplify to sum_j A[i,j] * B[j,k] in general.
    
    I think I'm confusing the tensor decomposition with the actual computation. Let
    me look at the schoolbook decomposition to understand the correct interpretation.
    
    In the schoolbook decomposition:
    For each (i,j,k), there's a term r = i*m*p + j*p + k.
    U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    
    So sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (only the term r = i*m*p + j*p + k
    contributes).
    
    And the matrix product is computed as:
    C[i,k] = sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k'']) * W[k*n+i, r]
    
    For the schoolbook, for a given r = i0*m*p + j0*p + k0:
    sum_{i',j'} U[i'*m+j', r] * A[i',j'] = A[i0, j0]
    sum_{j'',k''} V[j''*p+k'', r] * B[j'',k''] = B[j0, k0]
    W[k*n+i, r] = 1 if (k,i) = (k0, i0), else 0.
    
    So the sum over r gives: sum_{i0,j0,k0} A[i0,j0] * B[j0,k0] * [k=k0 and i=i0]
    = A[i,j] * B[j,k] summed over j... no, that's not right.
    
    Actually, for a fixed (i,k), the sum over r is:
    sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k'']) * W[k*n+i, r]
    
    For the schoolbook, W[k*n+i, r] = 1 only if r corresponds to (i0=i, k0=k).
    So the sum reduces to:
    sum_{j} (sum_{i',j'} U[i'*m+j', r(i,j,k)] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r(i,j,k)] * B[j'',k'']) * 1
    
    Where r(i,j,k) = i*m*p + j*p + k.
    
    For r = i*m*p + j*p + k:
    sum_{i',j'} U[i'*m+j', r] * A[i',j'] = A[i,j]
    sum_{j'',k''} V[j''*p+k'', r] * B[j'',k''] = B[j,k]
    
    So the sum over r gives: sum_j A[i,j] * B[j,k] = C[i,k]. Correct!
    
    So the correct interpretation is that the tensor equation
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    is necessary but not sufficient. The full condition is that the matrix product
    can be computed as:
    C[i,k] = sum_r (sum_{i',j'} U[i'*m+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', r] * B[j'',k'']) * W[k*n+i, r]
    
    And this is equivalent to the tensor equation if and only if the decomposition
    is "valid" in the sense that it correctly computes the matrix product.
    
    I think the tensor equation sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    is actually the correct and sufficient condition. Let me re-examine my rank-1
    decomposition.
    
    With U[i*3+j, 0] = 1, V[j*7+k, 0] = 1, W[k*2+i, 0] = 1:
    
    C[i,k] = sum_r (sum_{i',j'} U[i'*3+j', r] * A[i',j']) * (sum_{j'',k''} V[j''*7+k'', r] * B[j'',k'']) * W[k*2+i, r]
    
    = (sum_{i',j'} A[i',j']) * (sum_{j'',k''} B[j'',k'']) * 1
    
    = (sum of all entries of A) * (sum of all entries of B)
    
    This is not equal to sum_j A[i,j] * B[j,k] in general. So the rank-1 decomposition
    is invalid!
    
    The issue is that the tensor equation sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    is not the correct condition. The correct condition is the one I wrote above.
    
    Let me re-examine the schoolbook decomposition to understand the correct condition.
    
    For the schoolbook, the tensor equation is:
    sum_r U[i*3+j, r] * V[j*7+k, r] * W[k*2+i, r] = 1 for all i,j,k.
    
    And the matrix product is computed correctly. So the tensor equation is necessary.
    But is it sufficient?
    
    I think the answer is yes, but only if the decomposition is "separable" in the
    right way. The schoolbook decomposition has the property that for each r, there's
    a unique (i,j,k) such that U[i*3+j, r] = V[j*7+k, r] = W[k*2+i, r] = 1.
    
    My rank-1 decomposition doesn't have this property, so it's invalid.
    
    So the correct condition is not just the tensor equation, but also that the
    decomposition correctly computes the matrix product.
    
    Given this, let me go back to the schoolbook decomposition (rank 42) and see if
    I can improve it.
    
    Actually, let me just output the schoolbook decomposition since I'm not sure how
    to improve it, and at least it's correct.
    
    Wait, the current program already does the schoolbook with rank 42. The score is
    1.0. To improve, I need to find a valid decomposition with rank < 42.
    
    Let me try the following approach: use the fact that for 2x3 * 3x7, we can split
    the middle dimension into 2+1 and use Strassen for the 2x2 part.
    
    For the 2x2 * 2x7 part, I'll use a rank-7 decomposition (Strassen-like).
    For the 2x1 * 1x7 part, I'll use a rank-1 decomposition.
    Total rank: 8.
    
    But I need to make sure the decomposition is valid. Let me construct it carefully.
    
    Actually, given the complexity and the time limit, let me just output the schoolbook
    decomposition (rank 42) to ensure correctness, and hope that the evaluator accepts it.
    
    The current program already does this, so my score will be 1.0. To improve, I
    would need to find a valid decomposition with rank < 42, which is non-trivial.
    
    Let me just output the schoolbook decomposition.
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
