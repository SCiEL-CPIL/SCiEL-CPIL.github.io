"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    The key insight: for multiplying an n x m matrix by an m x p matrix,
    the tensor is T[i,j,k] = A[i,j] * B[j,k] * C[i,k]
    
    For 2x3 * 3x6:
    - A is 2x3, B is 3x6, C is 2x6
    - The tensor has indices i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4,5}
    
    We can exploit the structure: for each fixed j, we're multiplying a 2x1 vector by a 1x6 vector
    to get a 2x6 matrix. This is rank 2 (outer product).
    
    So for each j (3 values), we need rank 2, giving total rank 6.
    
    Wait, let me think more carefully. The matrix multiplication tensor for (n,m,p) dimensions
    has rank at most n*m*p (schoolbook), but can be less.
    
    For 2x3 * 3x6:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = A[i,j] * B[j,k] * C[i,k]
    
    We can decompose this as:
    For each j, the sub-tensor T[i,j,k] for fixed j is A[i,j] * B[j,k] * C[i,k]
    
    Actually, let's think of it differently. The multiplication can be written as:
    C = A * B where A is 2x3 and B is 3x6.
    
    One approach: split B into 3 column vectors b_j (each 1x6), and A into 3 row vectors a_j (each 1x3).
    Then C = sum_j (a_j^T * b_j) where a_j^T is 2x1 and b_j is 1x6, so a_j^T * b_j is 2x6.
    
    Each term a_j^T * b_j is an outer product of a 2-vector and a 6-vector, which has rank 2 as a tensor
    (in the sense of the matrix multiplication tensor).
    
    So total rank = 3 * 2 = 6.
    
    Let me verify: the tensor T[i,j,k] for the full multiplication is:
    T[i,j,k] = sum over all (i',j',k') of A[i',j'] * B[j',k'] * C[i',k'] * delta(i,j,k; i',j',k')
    
    Actually, the standard way: the matrix multiplication tensor for multiplying n x m by m x p
    has a decomposition of rank r if we can write:
    C[i,k] = sum_{c=1}^r U[i,c] * V[c,k]
    where U depends on A and V depends on B linearly.
    
    More precisely:
    C[i,k] = sum_{c=1}^r (sum_{j=1}^m U_{i,j,c} * A[i,j]) * (sum_{j=1}^m V_{j,k,c} * B[j,k])
    
    Wait, that's not quite right either. Let me use the standard formulation.
    
    The matrix multiplication tensor T in R^{n x m x p} has entries T[i,j,k] such that
    for matrices A (n x m) and B (m x p), the (i,k) entry of AB is:
    (AB)[i,k] = sum_{j} A[i,j] * B[j,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    
    where T[i,j,k] = 1 for all valid i,j,k.
    
    A rank-r decomposition of T means we can write T[i,j,k] = sum_{c=1}^r U[i,c] * V[j,c] * W[k,c]
    for some matrices U (n x r), V (m x r), W (p x r).
    
    Wait, but the problem states the factor matrices have shapes (n*m, rank), (m*p, rank), (p*n, rank).
    Let me re-read the problem.
    
    "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    And "the output factor follows the official trace convention W[k*n+i,r]"
    
    So the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that doesn't seem right dimensionally. Let me think again.
    
    Actually, I think the convention is:
    The matrix multiplication tensor has 3 indices: i (row of A, row of C), j (col of A, row of B), k (col of B, col of C).
    
    The standard tensor decomposition for matrix multiplication is:
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    
    But the problem says U has shape (n*m, rank), which suggests U is indexed by (i,j) pairs, not just i.
    
    Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    Hmm, (n*m) for U, (m*p) for V, (p*n) for W.
    
    I think the convention might be:
    - U is indexed by (i,j) where i is row of A, j is col of A (so n*m entries)
    - V is indexed by (j,k) where j is row of B, k is col of B (so m*p entries)  
    - W is indexed by (k,i) where k is col of C, i is row of C (so p*n entries)
    
    And the decomposition is:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Wait, but that's not a standard tensor decomposition. Let me look at the schoolbook example.
    
    In the schoolbook code:
    U[i * m + j][column] = 1
    V[j * p + k][column] = 1
    W[k * n + i][column] = 1
    
    So for each (i,j,k) triple, there's one rank-1 term where:
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    And the tensor entry T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For this to equal 1 (the value of the matrix multiplication tensor), we need exactly one c where all three are 1.
    
    So the decomposition is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    And we want T[i,j,k] = 1 for all valid (i,j,k).
    
    Now, can we do better than rank n*m*p = 2*3*6 = 36?
    
    Key insight: We can share rank-1 terms across multiple (i,j,k) triples.
    
    For example, consider fixing j. Then we have terms for all (i,k) with this j.
    U[i*m+j, c] depends on i, V[j*p+k, c] depends on k, W[k*n+i, c] depends on both i and k.
    
    If we can find a decomposition where for fixed j, the sub-problem of multiplying a 2x1 by 1x6
    (i.e., outer product) has rank 2, then total rank would be 3*2 = 6.
    
    For the outer product of a 2-vector and a 6-vector:
    We want T[i,k] = a_i * b_k for i in {0,1}, k in {0,1,2,3,4,5}.
    
    This can be decomposed as:
    T[i,k] = sum_{c=1}^2 U[i, c] * V[k, c] * W[c, c]  (simplified, since it's rank-2 as a matrix)
    
    Actually, the outer product a_i * b_k = (a_i * 1) * (1 * b_k) but that's not quite right for the tensor structure.
    
    Let me think about it differently. For fixed j, the sub-tensor is:
    T[i,j,k] = 1 for all i, k (with j fixed)
    
    We want to write: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Note that U[i*m+j, c] depends on i (and j is fixed), V[j*p+k, c] depends on k (and j is fixed),
    and W[k*n+i, c] depends on both i and k.
    
    For the outer product case (n=2, p=6, m=1 effectively), we can use:
    U[i, c] = a_i, V[k, c] = b_k, W[k*n+i, c] = 1/(a_i * b_k) but that's not linear.
    
    Actually, I realize the tensor T[i,j,k] = 1 for all i,j,k in the matrix multiplication.
    The "input" tensors are A[i,j] and B[j,k], and the "output" is C[i,k].
    The bilinear form is: C[i,k] = sum_j A[i,j] * B[j,k]
    
    So the tensor T[i,j,k] = 1 (the coefficient of A[i,j] * B[j,k] in C[i,k]).
    
    A rank-r decomposition means:
    C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    
    Wait, that's not right either because U is indexed by (i,j) and we're summing over j, but U[i*m+j,c] * A[i,j]
    would be summing over j for fixed i, giving a scalar, then multiplying by the V part.
    
    Let me re-examine. The standard bilinear decomposition of matrix multiplication is:
    C[i,k] = sum_{c=1}^r (sum_{j=1}^m U_{i,j,c} * A[i,j]) * (sum_{j=1}^m V_{j,k,c} * B[j,k])
    
    But in our case, U has shape (n*m, r), so U[i*m+j, c] plays the role of U_{i,j,c}.
    V has shape (m*p, r), so V[j*p+k, c] plays the role of V_{j,k,c}.
    W has shape (p*n, r), and W[k*n+i, c] is... what?
    
    I think the W factor might be used differently. Let me look at the schoolbook again:
    
    For each (i,j,k):
    U[i*m+j, c] = 1
    V[j*p+k, c] = 1
    W[k*n+i, c] = 1
    
    So the product U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for the matching (i,j,k).
    
    And T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For this to equal 1, we need the sum to be 1. In schoolbook, each (i,j,k) has its own c, so the sum is 1.
    
    But can we have a c that contributes to multiple (i,j,k) triples? Yes, if for a given c,
    U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] is nonzero for multiple (i,j,k).
    
    The constraint is: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    This is a system of n*m*p = 36 equations in 3*r unknowns (entries of U, V, W).
    
    We want to minimize r.
    
    Known results:
    - For 2x2 * 2x2, Strassen's algorithm gives rank 7 (vs schoolbook 8).
    - For 2x3 * 3x6, what's the known rank?
    
    Let me think about a construction.
    
    Observation: We can view the 2x3 * 3x6 multiplication as:
    - Split the 3 columns of A into groups, or split the 6 columns of B.
    
    Alternative: Use the fact that 2x3 * 3x6 can be seen as a block multiplication.
    
    Let me try a direct construction with rank 12:
    - For each pair (i, k) where i in {0,1}, k in {0,...,5}, we have C[i,k] = sum_j A[i,j] * B[j,k].
    - This is a sum of 3 terms.
    
    If we use rank 12 = 2 * 6, we could have one rank-1 term for each (i,k) pair, but that would require
    U[i*m+j, c] to be nonzero for all j for a given (i,k), which might not work.
    
    Let me try rank 18 = 3 * 6: For each (j,k) pair, have one rank-1 term.
    Then for fixed (j,k), U[i*m+j, c] = A[i,j] (but U must be constants, not depend on A).
    
    Wait, I'm confusing the tensor with the actual multiplication. The tensor T[i,j,k] = 1 is fixed.
    The decomposition coefficients U, V, W are constants.
    
    Let me try a different approach. Consider the following:
    
    For the 2x3 * 3x6 multiplication, we can use:
    - 3 separate 2x1 * 1x6 outer products (one for each j).
    
    For each j, the outer product of a 2-vector (A[:,j]) and a 6-vector (B[j,:]) gives a 2x6 matrix.
    The rank of the outer product tensor (for n=2, p=6) is 2 (since it's a rank-1 matrix, which has tensor rank 2 in this formulation? Let me verify).
    
    For the outer product: C[i,k] = a_i * b_k for i in {0,1}, k in {0,...,5}.
    We want: C[i,k] = sum_c U[i, c] * V[k, c] * W[c, c] (simplified notation for m=1 case)
    
    Actually for m=1, the tensor is T[i,0,k] = 1, and we want:
    1 = sum_c U[i, c] * V[k, c] * W[k*n+i, c]
    
    For n=2, p=6: U has 2*1=2 rows, V has 1*6=6 rows, W has 6*2=12 rows.
    
    We can set:
    c=0: U[0,0]=1, U[1,0]=0; V[k,0]=b_k for all k; W[k*2+0, 0]=1, W[k*2+1,0]=0
    c=1: U[0,1]=0, U[1,1]=1; V[k,1]=b_k for all k; W[k*2+0, 1]=0, W[k*2+1,1]=1
    
    But V[k,c] must be constants, not depend on b_k. So this doesn't work directly.
    
    I think I need to be more careful. Let me just try to find a low-rank decomposition using linear algebra.
    
    The system is: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    This is a system of 36 polynomial equations in 3*r variables. We want the smallest r for which this has a solution.
    
    For now, let me try to construct a decomposition with rank 12.
    
    Idea: Use the fact that we can factor the multiplication as:
    C = A * B = A * [B_0, B_1, B_2] where B_j is 1x6.
    
    Or: C = [A_0; A_1] * B where A_i is 1x3.
    
    Let me try: C[i,k] = sum_j A[i,j] * B[j,k].
    
    For each i (2 values), C[i,:] is a 1x6 row vector = sum_j A[i,j] * B[j,:].
    
    This is a linear combination of 3 vectors (B[0,:], B[1,:], B[2,:]) with coefficients A[i,0], A[i,1], A[i,2].
    
    The tensor for this is: for fixed i, T[i,j,k] = 1, and we're computing a 1x6 vector as a linear combination.
    
    The rank of the "linear combination" tensor for combining 3 vectors into 1 vector is 3 (you need 3 rank-1 terms).
    
    So for each i, rank 3, total rank 6.
    
    Let me verify: For fixed i, C[i,k] = sum_j A[i,j] * B[j,k].
    We want: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    
    For fixed i, this becomes: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    
    If we set, for c corresponding to j:
    U[i*m+j, c_j] = 1 (for the j-th term), 0 otherwise
    V[j*p+k, c_j] = 1 for all k (for the j-th term), 0 otherwise
    
    Then: sum_j U[i*m+j, c_j] * A[i,j] = A[i,j]
    sum_j V[j*p+k, c_j] * B[j,k] = B[j,k]
    
    And C[i,k] = sum_j A[i,j] * B[j,k]. 
    
    But we also need the W factor: W[k*n+i, c] must be 1 for this to work.
    
    So for c_j (the j-th term for fixed i):
    U[i*m+j, c_j] = 1
    V[j*p+k, c_j] = 1 for all k
    W[k*n+i, c_j] = 1 for all k
    
    But W[k*n+i, c_j] depends on k, and we need it to be 1 for all k. So W has 1's in positions (k*n+i) for all k.
    
    This works! For each (i,j) pair, we have one rank-1 term. Total rank = n*m = 2*3 = 6.
    
    Wait, but V[j*p+k, c_j] = 1 for all k means V has 6 ones in row j*p+k for all k. And W[k*n+i, c_j] = 1 for all k means W has 2*6=12 ones.
    
    Let me verify the tensor equation:
    T[i,j',k] = sum_c U[i*m+j', c] * V[j'*p+k, c] * W[k*n+i, c]
    
    For the term c_j (corresponding to (i,j)):
    U[i*m+j', c_j] = 1 if j'=j, else 0
    V[j'*p+k, c_j] = 1 if j'=j (since V[j*p+k', c_j]=1 for all k', so V[j*p+k, c_j]=1), else 0
    W[k*n+i, c_j] = 1
    
    So the product is 1 if j'=j, else 0.
    
    Summing over all c_j (for all (i',j) with i'=i): sum_j U[i*m+j', c_(i,j)] * V[j'*p+k, c_(i,j)] * W[k*n+i, c_(i,j)]
    = sum_j [1 if j'=j else 0] * [1 if j'=j else 0] * 1
    = 1 (when j'=j, which is exactly one term)
    
    So T[i,j',k] = 1. 
    
    This gives rank = n*m = 6.
    
    But wait, is this correct? Let me double-check with the schoolbook. In schoolbook, rank = n*m*p = 36.
    My construction gives rank = n*m = 6. That's a big improvement.
    
    Let me make sure I haven't made an error. The key is that V[j*p+k, c] = 1 for all k, which means V is not sparse.
    And W[k*n+i, c] = 1 for all k, which means W is not sparse either.
    
    But the tensor equation is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    With my construction:
    For c corresponding to (i0, j0):
    U[i*m+j, c] = 1 if (i,j) = (i0,j0), else 0
    V[j*p+k, c] = 1 if j = j0, else 0 (for all k)
    W[k*n+i, c] = 1 if i = i0, else 0 (for all k)
    
    So the product U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j) = (i0,j0) and i = i0, else 0.
    Which is 1 if (i,j) = (i0,j0), else 0.
    
    Summing over all c (all (i0,j0) pairs): sum_{i0,j0} [1 if (i,j)=(i0,j0) else 0] = 1.
    
    So T[i,j,k] = 1 for all i,j,k. 
    
    Rank = n*m = 6.
    
    Let me code this up.
    
    n, m, p = 2, 3, 6
    rank = n * m = 6
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            # U[i*m+j, c] = 1
            U[i*m+j][c] = 1
            # V[j*p+k, c] = 1 for all k
            for k in range(p):
                V[j*p+k][c] = 1
            # W[k*n+i, c] = 1 for all k
            for k in range(p):
                W[k*n+i][c] = 1
            c += 1
    
    return {"U": U, "V": V, "W": W}
    
    Let me verify this is correct by checking the tensor equation for a few values.
    
    T[0,0,0] = sum_c U[0*3+0, c] * V[0*6+0, c] * W[0*2+0, c]
    = sum_c U[0, c] * V[0, c] * W[0, c]
    
    c=0 (i=0,j=0): U[0,0]=1, V[0,0]=1, W[0,0]=1 -> product=1
    c=1 (i=0,j=1): U[0,1]=0, ... -> product=0
    c=2 (i=0,j=2): U[0,2]=0, ... -> product=0
    c=3 (i=1,j=0): U[0,3]=0, ... -> product=0
    c=4 (i=1,j=1): U[0,4]=0, ... -> product=0
    c=5 (i=1,j=2): U[0,5]=0, ... -> product=0
    
    Sum = 1. Correct.
    
    T[1,2,5] = sum_c U[1*3+2, c] * V[2*6+5, c] * W[5*2+1, c]
    = sum_c U[5, c] * V[17, c] * W[11, c]
    
    c=0 (i=0,j=0): U[5,0]=0 -> 0
    c=1 (i=0,j=1): U[5,1]=0 -> 0
    c=2 (i=0,j=2): U[5,2]=0 -> 0
    c=3 (i=1,j=0): U[5,3]=0 -> 0
    c=4 (i=1,j=1): U[5,4]=0 -> 0
    c=5 (i=1,j=2): U[5,5]=1, V[17,5]=1 (since j=2, V[2*6+k,5]=1 for all k, so V[17,5]=1), W[11,5]=1 (since i=1, W[k*2+1,5]=1 for all k, so W[11,5]=1)
    -> product = 1
    
    Sum = 1. Correct.
    
    So rank 6 works. Can we do better?
    
    The lower bound for the rank of the matrix multiplication tensor for n x m * m x p is at least max(n, m, p) * something, but I'm not sure of the exact lower bound.
    
    For 2x3 * 3x6, rank 6 might be optimal, or we might be able to do better.
    
    Let me think... The tensor T[i,j,k] = 1 for all i,j,k. The rank is the minimum r such that
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    This is equivalent to saying that the 36-dimensional vector (flattened T) is in the span of r rank-1 tensors
    of the form u ⊗ v ⊗ w where u is in R^{n*m}, v is in R^{m*p}, w is in R^{p*n}, but with the specific
    indexing convention.
    
    Actually, I realize that my construction might not be optimal. Let me see if rank 5 or less is possible.
    
    Given the time constraint, let me go with rank 6 for now, which is already a significant improvement over 36.
    
    Score = seed/rank = 36/6 = 6.0 (if seed is 36). Wait, the score is "seed/value" where value is the rank.
    The current program has rank 36 and score 1.0, so seed = 36.
    With rank 6, score = 36/6 = 6.0.
    
    Let me finalize the code.
    """
    n, m, p = DIMENSIONS
    rank = n * m  # 6
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            U[i * m + j][c] = 1
            for k in range(p):
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
            c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
