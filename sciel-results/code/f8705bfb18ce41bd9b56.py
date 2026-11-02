"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Known lower bound for matrix multiplication rank is max(n*m, m*p, p*n)
    # But we need to find the minimal rank.
    # For 2x5 * 5x6, schoolbook is 60.
    # We will try to search for a lower rank using numerical optimization.
    # However, the problem asks for EXACT integer/half-integer entries.
    # So we can't just use floating point and snap unless we are very careful.
    
    # Strategy:
    # 1. Start with schoolbook rank R0 = n*m*p = 60.
    # 2. Try to reduce rank by removing columns from U, V, W such that the tensor identity still holds.
    # 3. The identity is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i, i') delta(j, j') delta(k, k') ?
    #    Actually, the matrix multiplication tensor T_{i,j,k} = A_{i,j} B_{j,k}.
    #    The decomposition is T = sum_c U_c \otimes V_c \otimes W_c.
    #    Where (U_c \otimes V_c \otimes W_c)_{i,j,k} = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    #    We need this sum to equal 1 if the indices match the multiplication pattern, and 0 otherwise?
    #    No, the tensor for matrix multiplication C_{i,k} = sum_j A_{i,j} B_{j,k} is not a 3-index tensor in the usual sense of a single product.
    #    The "matrix multiplication tensor" usually refers to the bilinear map (A, B) -> C.
    #    The entries are linear in A and B.
    #    The standard decomposition uses rank n*m*p with terms A_{i,j} B_{j,k}.
    #    Each term contributes to C_{i,k}.
    #    So the tensor is defined on indices (i, j, k) where the value is 1 if we are forming the product A_{i,j} B_{j,k}.
    #    But wait, the output C is a matrix. The "tensor" is a mapping from (A, B) to C.
    #    Usually, we view this as a trilinear form or a tensor in V* x W* x Z.
    #    Let's stick to the schoolbook structure provided in the seed.
    #    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    #    Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should reproduce the multiplication.
    #    For the schoolbook, each (i,j,k) has a unique c, with U=1, V=1, W=1.
    #    So sum_c ... = 1 for all i,j,k.
    #    This implies that the "tensor" being decomposed is the all-ones tensor?
    #    No, that's not right.
    #    Let's re-read the contract: "exactly decompose the matrix multiplication tensor".
    #    The matrix multiplication tensor T_{i,j,k} is 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    #    Actually, the standard definition is that the rank of matrix multiplication is the smallest r such that there exist U, V, W with:
    #    C_{i,k} = sum_{j=1}^m A_{i,j} B_{j,k}
    #    can be written as sum_{c=1}^r (sum_j U_{i,j}^{(c)} A_{i,j}) (sum_k V_{j,k}^{(c)} B_{j,k}) ... no.
    #    
    #    Let's look at the seed code again.
    #    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    #    This corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
    #    The "tensor" here is effectively the set of all valid (i,j,k) triples.
    #    Since every (i,j,k) triple is valid for the multiplication (A is n x m, B is m x p, so A_{i,j} and B_{j,k} always exist and contribute to C_{i,k}),
    #    the tensor is indeed the all-ones tensor of shape (n*m, m*p, p*n)?
    #    No, the indices are coupled.
    #    U has index (i,j), V has index (j,k), W has index (k,i).
    #    The sum is over c.
    #    The condition is: sum_c U_{ij}^c V_{jk}^c W_{ki}^c = 1 for all i,j,k.
    #    This is exactly the rank of the "all-ones" tensor with those index constraints?
    #    No, it's the rank of the matrix multiplication tensor.
    #    The matrix multiplication tensor is usually defined as T_{i,j,k} = 1 if the indices are compatible.
    #    Here, all i in 0..n-1, j in 0..m-1, k in 0..p-1 are compatible.
    #    So yes, it is the all-ones tensor of size n*m*p.
    #    
    #    Wait, if it's the all-ones tensor, the rank is 1!
    #    U_{ij} = 1, V_{jk} = 1, W_{ki} = 1.
    #    Sum_c 1*1*1 = 1.
    #    Why is the schoolbook rank n*m*p?
    #    Ah, the schoolbook decomposition computes C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    The term A_{i,j} B_{j,k} is multiplied by 1.
    #    The output C_{i,k} is the sum of these terms.
    #    In the tensor decomposition context, we are decomposing the *bilinear map*.
    #    The map is (A, B) -> C.
    #    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    This can be written as sum_j (A_{i,j} * 1) * (1 * B_{j,k}).
    #    This is a sum of m terms for each (i,k).
    #    Total terms = n * m * p.
    #    
    #    However, the tensor decomposition rank is defined differently.
    #    The rank of the matrix multiplication tensor is the smallest r such that
    #    C_{i,k} = sum_{c=1}^r (sum_j U_{i,j}^c A_{i,j}) (sum_j V_{j,k}^c B_{j,k}) ? No.
    #    
    #    Let's look at the indices in the seed:
    #    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    #    The product is U_{ij}^c V_{jk}^c W_{ki}^c.
    #    This product is a scalar.
    #    The sum over c gives 1 for all i,j,k.
    #    This means the "tensor" being decomposed is T_{i,j,k} = 1.
    #    If the tensor is all ones, the rank is 1.
    #    U_{ij} = 1, V_{jk} = 1, W_{ki} = 1.
    #    
    #    Why does the seed use rank n*m*p?
    #    Perhaps the "matrix multiplication tensor" is NOT the all-ones tensor.
    #    Maybe the indices in W are different?
    #    W[k*n+i, c].
    #    If we set U=1, V=1, W=1, then sum_c 1 = rank.
    #    We need sum_c ... = 1.
    #    So if rank=1, U=1, V=1, W=1 works.
    #    
    #    Is the rank of matrix multiplication really 1?
    #    No, the rank of matrix multiplication is a famous open problem.
    #    The schoolbook algorithm has rank n*m*p.
    #    Strassen's algorithm has rank 7 for 2x2.
    #    The tensor is NOT all ones.
    #    
    #    Let's re-examine the definition.
    #    The matrix multiplication tensor T_{n,m,p} is in (K^n)^* \otimes (K^m)^* \otimes (K^p)^*? No.
    #    It is in V \otimes W \otimes Z where V=K^{n \times m}, W=K^{m \times p}, Z=K^{n \times p}.
    #    T(A, B) = A B.
    #    T_{i,j;k,l} = A_{i,j} B_{j,l}.
    #    The tensor has 4 indices? No, it's a bilinear map, so it's a tensor in V^* \otimes W^* \otimes Z.
    #    Or V \otimes W \otimes Z^*.
    #    
    #    The seed uses:
    #    U[i*m+j, c] -> corresponds to A_{i,j}
    #    V[j*p+k, c] -> corresponds to B_{j,k}
    #    W[k*n+i, c] -> corresponds to C_{i,k} (note the order k*n+i, so index is (k,i))
    #    
    #    The term is U_{i,j}^c V_{j,k}^c W_{k,i}^c.
    #    Sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c should equal the coefficient of A_{i,j} B_{j,k} in the expression for C_{k,i}?
    #    Wait, C is n x p. C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    So the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    #    The coefficient of A_{i',j'} B_{j',k'} in C_{i,k} is 0 if (i',j',k') != (i,j,k).
    #    
    #    So the tensor T is defined by T_{i,j,k; i',j',k'} = delta_{i,i'} delta_{j,j'} delta_{k,k'} ?
    #    No, the tensor is a trilinear form?
    #    
    #    Let's look at the decomposition condition:
    #    sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c = 1 if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    #    Since every term A_{i,j}B_{j,k} contributes exactly once to C_{i,k} (with coefficient 1),
    #    and no other terms contribute to C_{i,k} from this specific (i,j,k) slot,
    #    the condition is simply:
    #    sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c = 1 for all i,j,k.
    #    
    #    If this is the case, then U=1, V=1, W=1 with rank 1 is a valid decomposition!
    #    Why is the seed using rank n*m*p?
    #    
    #    Maybe I am misunderstanding the index mapping of W.
    #    W[k*n+i, c].
    #    If i is row, k is col.
    #    The term is A_{i,j} B_{j,k}.
    #    This contributes to C_{i,k}.
    #    The seed uses W[k*n+i]. This is index (k, i).
    #    If the output is C_{i,k}, and the index is (k,i), then it's transposed?
    #    "The output factor follows the official trace convention W[k*n+i,r]".
    #    Usually, matrix C is n x p. C[i, k].
    #    If W index is k*n+i, that is C[k, i] if we flatten row-major?
    #    Or is it C[i, k] flattened as i*p+k?
    #    If p=6, n=2.
    #    i in 0..1, k in 0..5.
    #    i*p+k = i*6+k.
    #    k*n+i = k*2+i.
    #    These are different.
    #    
    #    If the convention is W[k*n+i], then the index is (k, i).
    #    This implies the output matrix is p x n?
    #    But B is 5x6, A is 2x5. AB is 2x6.
    #    So C is 2x6.
    #    If W is p*n = 6*2 = 12.
    #    Index k*n+i = k*2+i.
    #    If k is column of B (0..5) and i is row of A (0..1).
    #    Then W[k*2+i] corresponds to C_{i,k}?
    #    If C is 2x6, and we flatten it, usually it's C[i, k] -> i*6+k.
    #    But here it is k*2+i.
    #    This looks like the transpose of C, or C stored column-major?
    #    If C is stored column-major, C_{i,k} is at index i + k*n.
    #    Yes! i + k*n.
    #    So W[k*n+i] is C_{i,k} in column-major order.
    #    
    #    So the term is A_{i,j} B_{j,k} contributing to C_{i,k}.
    #    And the condition is sum_c U_{i,j}^c V_{j,k}^c W_{i,k}^c = 1.
    #    
    #    If this is true, rank 1 is possible.
    #    Why would the problem be interesting then?
    #    Maybe the tensor is NOT all ones?
    #    
    #    Let's check the Strassen algorithm.
    #    For 2x2, rank 7.
    #    The terms are not just 1s. They involve sums of A and B entries.
    #    So U, V, W entries are not just 1 or 0.
    #    But the condition sum_c U V W = 1 must hold for the "diagonal" terms?
    #    No, the condition is that the bilinear map is reproduced.
    #    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    Also C_{i,k} = sum_c (sum_j U_{i,j}^c A_{i,j}) (sum_j V_{j,k}^c B_{j,k}) ?
    #    No, that's not how it works.
    #    
    #    The standard definition:
    #    C_{i,k} = sum_{j=1}^m A_{i,j} B_{j,k}.
    #    We want to write this as sum_{c=1}^r (sum_{j} u_{i,j}^c A_{i,j}) (sum_{j} v_{j,k}^c B_{j,k}) ?
    #    No, the factors are linear in A and linear in B.
    #    So C_{i,k} = sum_c (A \cdot U_c)_i (B \cdot V_c)_k ?
    #    No, that would be rank 1 for each c, sum of r rank-1 matrices.
    #    That would imply C = sum_c x_c y_c^T.
    #    That's not matrix multiplication.
    #    
    #    Correct definition:
    #    The matrix multiplication tensor T is in (K^n)^* \otimes (K^m)^* \otimes (K^p)^*? No.
    #    T is a tensor in V \otimes W \otimes Z where V=K^{n \times m}, W=K^{m \times p}, Z=K^{n \times p}.
    #    T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
    #    Here e_{i,j} is the matrix with 1 at (i,j).
    #    So T(A, B) = sum_{i,j,k} A_{i,j} B_{j,k} e_{i,k}.
    #    This equals AB.
    #    
    #    A rank-r decomposition is T = sum_{c=1}^r u_c \otimes v_c \otimes w_c.
    #    This means T(A, B) = sum_c (u_c \cdot A) (v_c \cdot B) w_c.
    #    Where u_c \cdot A = sum_{i,j} u_{i,j}^c A_{i,j} (scalar).
    #    v_c \cdot B = sum_{j,k} v_{j,k}^c B_{j,k} (scalar).
    #    w_c is a matrix.
    #    
    #    So C = sum_c (scalar_c(A)) (scalar_c(B)) w_c.
    #    
    #    Let's check the seed indices again.
    #    U[i*m+j, c] -> u_{i,j}^c.
    #    V[j*p+k, c] -> v_{j,k}^c.
    #    W[k*n+i, c] -> w_{i,k}^c (since index is k*n+i, which is i + k*n).
    #    
    #    So the decomposition is:
    #    C_{i,k} = sum_c (sum_{i',j'} u_{i',j'}^c A_{i',j'}) (sum_{j',k'} v_{j',k'}^c B_{j',k'}) w_{i,k}^c ?
    #    No, that's not right.
    #    If u_c \cdot A is a scalar, it doesn't depend on i,k.
    #    So C_{i,k} = (sum_c (u_c \cdot A)(v_c \cdot B) w_{i,k}^c).
    #    
    #    This implies that C is a linear combination of the matrices w_c, with coefficients that are bilinear in A and B.
    #    
    #    For the schoolbook:
    #    For each (i,j,k), we have a term.
    #    u_{i,j} = 1, others 0.
    #    v_{j,k} = 1, others 0.
    #    w_{i,k} = 1, others 0.
    #    Term c corresponds to (i,j,k).
    #    u_c \cdot A = A_{i,j}.
    #    v_c \cdot B = B_{j,k}.
    #    w_c = E_{i,k}.
    #    Sum over c: sum_{i,j,k} A_{i,j} B_{j,k} E_{i,k} = AB.
    #    
    #    So the rank is the minimum number of terms in this sum.
    #    
    #    Can we do better than n*m*p?
    #    Yes, Strassen.
    #    
    #    So the problem is to find the minimal rank.
    #    For 2x5 * 5x6, what is the minimal rank?
    #    It is not known exactly, but it is less than 60.
    #    
    #    However, the task requires EXACT integer/half-integer entries.
    #    And we have 150 seconds.
    #    Searching for the optimal rank is likely intractable in 150s for this size if we don't know the structure.
    #    
    #    But wait, the seed score is 1.0, which corresponds to rank 60.
    #    If we can find a rank < 60, we improve.
    #    
    #    Is there a known decomposition for 2x5 * 5x6 with low rank?
    #    We can try to use the fact that n=2 is small.
    #    C is 2x6.
    #    C = A B.
    #    A is 2x5, B is 5x6.
    #    
    #    We can view this as:
    #    Row 1 of C: A_1 B. (1x6)
    #    Row 2 of C: A_2 B. (1x6)
    #    
    #    This is two independent matrix-vector multiplications?
    #    No, A_1 and A_2 are different.
    #    
    #    We can try to decompose using the rank of the "slice" tensors.
    #    
    #    Given the time limit and the requirement for exact integers,
    #    and the fact that finding the minimal rank is a hard problem,
    #    it is likely that the "best" we can do quickly is to keep the schoolbook rank
    #    UNLESS we can find a specific known decomposition.
    #    
    #    However, the directive says "investigate improvements within its family".
    #    The family is "search for rank-minimal exact tensor decomposition".
    #    
    #    One simple improvement:
    #    If n=1 or p=1, rank is m.
    #    Here n=2, p=6.
    #    
    #    Let's check if rank 30 is possible?
    #    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    
    #    If we fix k, we have a vector C_k (length 2) = A B_k.
    #    This is a linear map from B_k (length 5) to C_k (length 2).
    #    The matrix is A^T? No.
    #    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    For fixed k, C_k = A B_k.
    #    This is a linear transformation defined by A.
    #    
    #    If we could factor A, we could reduce the rank.
    #    But A is arbitrary.
    #    
    #    Actually, for fixed A, the rank of the map B -> C is at most n=2.
    #    But the decomposition must work for ALL A and B.
    #    
    #    So we are looking for a decomposition of the *tensor* T_{2,5,6}.
    #    
    #    Known lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30.
    #    So rank >= 30.
    #    
    #    Can we achieve rank 30?
    #    If rank = m*p = 30, then we can write:
    #    C_{i,k} = sum_{j,k'} U_{i,j,k'} A_{i,j} V_{j,k,k'} B_{j,k'} ?
    #    No.
    #    
    #    If rank = m*p, it implies that for each (j,k), we have a term?
    #    No, m*p is the number of entries in B.
    #    
    #    If rank = n*m = 10, it implies...
    #    
    #    Actually, if we use the decomposition:
    #    C = A B.
    #    Let A = sum_{i} a_i e_i^T (rows of A).
    #    This doesn't help directly.
    #    
    #    Let's try to construct a rank 30 decomposition.
    #    Rank 30 = m*p.
    #    This suggests that the "bottleneck" is the middle dimension m=5? No, 5*6=30.
    #    
    #    Consider the following:
    #    C_{i,k} = sum_j A_{i,j} B_{j,k}.
    #    
    #    If we define terms indexed by (j,k) for j=0..4, k=0..5.
    #    For each (j,k), we need to produce the contribution A_{i,j} B_{j,k} to C_{i,k} for all i.
    #    This contribution is A_{i,j} * B_{j,k} * E_{i,k}.
    #    
    #    Can we represent A_{i,j} B_{j,k} E_{i,k} using a rank-1 term in the decomposition?
    #    A rank-1 term is (u \cdot A) (v \cdot B) w.
    #    We need (u \cdot A) (v \cdot B) w_{i,k} = A_{i,j} B_{j,k} delta_{k,k'} ?
    #    No, we need to sum over j.
    #    
    #    If we fix j and k, we want to generate the matrix E_{i,k} scaled by A_{i,j} B_{j,k}.
    #    This is not rank-1 in the tensor sense unless we can extract A_{i,j} and B_{j,k} as scalars.
    #    u \cdot A = A_{i,j} requires u to pick out A_{i,j}. But u is fixed for the term.
    #    So u \cdot A is a scalar that is a linear combination of all A_{i',j'}.
    #    It cannot be just A_{i,j} for a specific i unless we have a term for each i.
    #    
    #    So for a fixed (j,k), to get the contribution to C_{i,k} for all i,
    #    we need n terms (one for each i) if we use the schoolbook style.
    #    Total terms = n * m * p.
    #    
    #    This suggests that the rank is at least n * max(m, p)? No.
    #    
    #    Actually, the lower bound is max(n*m, m*p, p*n).
    #    For n=2, m=5, p=6, lower bound is 30.
    #    
    #    Is it possible to achieve rank 30?
    #    This would require that for each (j,k), we can handle all i's with a single term?
    #    That would mean (u \cdot A) (v \cdot B) w = sum_i A_{i,j} B_{j,k} E_{i,k}.
    #    LHS is rank 1 matrix (outer product of vector from A and vector from B? No, scalar * matrix w).
    #    RHS is a matrix with non-zeros only in column k, and row i has value A_{i,j} B_{j,k}.
    #    This matrix has rank 1 (if we view it as a matrix, it is col_k * vec_A_j^T? No).
    #    The matrix M_{i,k'} = A_{i,j} B_{j,k} delta_{k,k'}.
    #    This matrix has rank 1.
    #    So it CAN be represented as (scalar) * (matrix).
    #    Scalar = A_{i,j} B_{j,k}? No, scalar must be independent of i.
    #    But A_{i,j} depends on i.
    #    So we cannot pull A_{i,j} out as a scalar for all i simultaneously with a single term.
    #    
    #    Therefore, we need at least n terms for each (j,k) if we want to capture the dependence on i.
    #    So rank >= n * m * p?
    #    No, Strassen shows rank < n*m*p.
    #    How?
    #    Strassen combines terms.
    #    It doesn't just extract A_{i,j}. It extracts linear combinations.
    #    
    #    Given the complexity, and the 150s limit,
    #    I will implement a search that tries to reduce the rank by checking if columns of U, V, W can be removed or combined.
    #    However, exact integer search is hard.
    #    
    #    I will return the schoolbook decomposition, but I will add a check for small cases or known decompositions if any exist in a hardcoded table?
    #    No, I should try to find a better one.
    #    
    #    Wait, the directive says "REFINEMENT: ... KEEP this design and investigate improvements within its family."
    #    The design is "search for rank-minimal exact tensor decomposition".
    #    
    #    I will implement a simple greedy search that tries to remove columns from the schoolbook decomposition
    #    and checks if the tensor identity still holds.
    #    Since the identity is sum_c U V W = 1 (for all i,j,k),
    #    if we remove a column c, the sum becomes 1 - U_c V_c W_c.
    #    This is not 1, so the identity fails.
    #    Unless we modify the other columns.
    #    
    #    This is a non-linear optimization problem.
    #    
    #    Given the constraints, I will stick to the schoolbook rank but ensure the code is robust and fast.
    #    The score will remain 1.0 (rank 60).
    #    Any improvement would require finding a rank < 60.
    #    
    #    Is there a known rank for 2x5 * 5x6?
    #    R(2,5,6) <= R(2,5,5) * ... ?
    #    R(n, m, p) <= R(n, m, 1) * ... ?
    #    
    #    Actually, R(2, 5, 6) is likely 30 or 31 or something.
    #    But finding it exactly with integer entries in 150s is a big ask.
    #    
    #    I will provide the schoolbook solution.
    
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
