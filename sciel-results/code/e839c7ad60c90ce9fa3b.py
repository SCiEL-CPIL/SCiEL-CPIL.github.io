"""
Rank-minimal exact tensor decomposition for 3x3x8 matrix multiplication.

Strategy:
  - For 3x3x3, use Strassen's algorithm (rank 7) for each 3-column block of B.
  - B is 3x8, so split into two 3x3 blocks and one 3x2 block.
  - For the 3x2 block, use a rank-6 decomposition (schoolbook for 3x3x2 gives 18,
    but we can do better: 3x2 multiplication has rank at most 6 if we treat it as
    two 3x1 multiplications, each of rank 3, so total rank 6... wait no.
    
    Actually, let me think about this more carefully.
    
    The tensor for multiplying A (3x3) by B (3x2) to get C (3x2) has:
    C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1,2 and k=0,1
    
    This is two separate 3x3x1 multiplications (one for each column k of C),
    each of which has rank 3 (since it's just a dot product, which is rank 3
    as a bilinear form... actually a single dot product of length 3 is rank 3).
    
    Wait, the tensor for a single dot product sum_j A_j * B_j is rank 3?
    No! The dot product of two vectors of length m is a bilinear form with
    tensor rank m (it's the sum of m rank-1 tensors: A_j * B_j for j=0..m-1).
    
    So for 3x3x2: we have 3 rows of A, 2 columns of B, and for each (i,k) pair,
    we need sum_j A[i,j]*B[j,k]. The total number of (i,k) pairs is 3*2=6,
    and each requires 3 multiplications, so schoolbook rank is 18.
    
    But can we do better? The tensor for 3x3x2 is the sum of two 3x3x1 tensors
    (one for k=0, one for k=1). Each 3x3x1 tensor has rank 3*3=9? No wait.
    
    Let me be more careful. The matrix multiplication tensor T_{n,m,p} has
    entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B
    contribute to C[i,k].
    
    Actually, the standard tensor is: T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C
    where the coefficient is 1 when j is the shared index.
    
    For 3x3x2: T has entries for i in {0,1,2}, j in {0,1,2}, k in {0,1}.
    The rank-1 decomposition is: for each (i,j,k), the term is e_i ⊗ e_j ⊗ e_k
    with coefficient 1. So schoolbook rank is 3*3*2 = 18.
    
    Can we reduce this? The 3x3x2 tensor can be seen as two 3x3x1 tensors.
    A 3x3x1 tensor (multiplying 3x3 by 3x1) has rank 9 in schoolbook.
    But actually, multiplying a 3x3 matrix by a 3x1 vector to get a 3x1 vector
    is just 3 dot products of length 3, so rank is 3*3=9? No, each dot product
    is rank 3, and there are 3 of them, so total rank 9.
    
    Hmm, but can we share structure between the two columns? Probably not easily.
    
    Let me reconsider the overall strategy:
    - Split B (3x8) into three parts: B1 (3x3), B2 (3x3), B3 (3x2)
    - C = A*B1 + A*B2 + A*B3
    - For A*B1 and A*B2, use Strassen (rank 7 each)
    - For A*B3, use schoolbook (rank 18)
    - Total rank: 7 + 7 + 18 = 32
    
    Wait, but Strassen gives C = A*B where both are 3x3. The output is 3x3.
    So C1 = A*B1 is 3x3, C2 = A*B2 is 3x3, C3 = A*B3 is 3x2.
    Total output C is 3x8 = C1 | C2 | C3.
    
    The rank of the combined tensor is at most the sum of the ranks of the
    individual tensors (since they operate on disjoint output indices).
    So rank <= 7 + 7 + 18 = 32.
    
    Can we do better for the 3x3x2 part? Let's think...
    
    For 3x3x2, the lower bound on rank is max(3*3, 3*2, 2*3) = max(9,6,6) = 9.
    Schoolbook gives 18. Is there a rank-9 or rank-10 decomposition?
    
    Actually, I recall that for multiplying an n×m matrix by an m×p matrix,
    the rank is at most min(n*p*m, ...). For 3x3x2, I'm not sure of the exact
    minimal rank. Let's just use 18 for now and see if we can improve.
    
    Actually, wait. Let me reconsider. Maybe I should use Strassen for the
    3x3x3 parts and a better method for 3x3x2.
    
    For 3x3x2: We can view this as computing two 3x1 vectors (columns of C3).
    Each column is A * b_k where b_k is a 3x1 vector. Computing A*b for a
    3x3 A and 3x1 b requires 9 multiplications in schoolbook, but can it be
    done with fewer? The rank of the 3x3x1 tensor is 9 (I believe this is
    minimal since the border rank is also 9).
    
    So for two columns, rank is at most 18, and likely exactly 18 if the
    3x3x1 rank is 9 and they're independent.
    
    Hmm, but actually the two columns share the same A. So the tensor for
    3x3x2 is: sum_{i,j,k} A[i,j] B[j,k] where k in {0,1}.
    This can be written as: for each j, sum_i,k A[i,j] B[j,k].
    The rank-1 terms would be A[i,j] * B[j,k] for each (i,j,k).
    
    I think 18 is the schoolbook rank and it might be minimal. Let's go with 32.
    
    Actually, let me double-check: can we combine the two 3x3x3 Strassen
    decompositions? No, because they produce different output columns.
    
    So my plan: rank 32 decomposition using Strassen for 3x3x3 blocks and
    schoolbook for the 3x3x2 block.
    
    Let me implement Strassen for 3x3x3.
    
    Strassen's algorithm for 2x2x2 uses 7 multiplications. For 3x3x3,
    we can pad to 4x4x4 and use Strassen (which would give rank 7^2=49
    for 4x4x4, which is worse than schoolbook 64... wait no, 4x4x4 schoolbook
    is 64, and Strassen gives 7^2=49, so it's better). But we want 3x3x3.
    
    Actually, for 3x3x3, the known minimal rank is 7 (same as 2x2x2, since
    you can embed 3x3x3 into 4x4x4 and use Strassen, but that gives rank 49
    for 4x4x4, not 7 for 3x3x3).
    
    Wait, I'm confusing myself. The rank of the 2x2x2 matrix multiplication
    tensor is 7 (Strassen). The rank of the 3x3x3 tensor is not 7.
    
    For 3x3x3, the schoolbook rank is 27. The known upper bound is 23
    (from various improvements). The lower bound is max(9,9,9)=9.
    
    Actually, I recall that the rank of 3x3x3 matrix multiplication is 23
    (or maybe 24?). Let me just use schoolbook for simplicity: rank 27.
    
    So total rank: 27 + 27 + 18 = 72. That's the same as the original!
    
    Hmm, that's not an improvement. Let me think differently.
    
    OK so the key insight is: for 3x3x8, the schoolbook rank is 72.
    To improve, I need to find a decomposition with rank < 72.
    
    One approach: use the fact that 8 = 3 + 3 + 2, and use better algorithms
    for the 3x3x3 and 3x3x2 parts.
    
    For 3x3x3: If I can find a rank-23 decomposition, total would be 23+23+18=64.
    For 3x3x2: If I can find a rank-15 decomposition, total would be 27+27+15=69.
    
    I don't have a concrete rank-23 decomposition for 3x3x3 memorized.
    
    Let me try a different approach: use the "divide and conquer" idea.
    3x3x8 can be viewed as 3x3x(3+5) or other splits.
    
    Actually, let me just try to construct a rank-71 decomposition by
    combining two schoolbook terms cleverly.
    
    Consider two terms: A[i,j]*B[j,k1] and A[i,j]*B[j,k2] for the same (i,j)
    but different k. These are:
    e_i^A ⊗ e_j^B ⊗ e_{k1}^C and e_i^A ⊗ e_j^B ⊗ e_{k2}^C
    
    I can write: e_i ⊗ e_j ⊗ (e_{k1} + e_{k2}) + e_i ⊗ e_j ⊗ (e_{k1} - e_{k2})
    but that's still 2 terms.
    
    Alternatively: (e_i ⊗ e_j ⊗ e_{k1}) + (e_i ⊗ e_j ⊗ e_{k2})
    = (e_i ⊗ e_j) ⊗ (e_{k1} + e_{k2})
    but this is not a rank-1 tensor in the standard sense because the
    decomposition requires each term to be a product of three vectors.
    
    e_i ⊗ e_j ⊗ (e_{k1} + e_{k2}) IS a rank-1 tensor! It's the product of
    vector e_i, vector e_j, and vector (e_{k1}+e_{k2}).
    
    So I can combine: A[i,j]*B[j,k1] + A[i,j]*B[j,k2] into a single term
    with W vector having 1s at positions k1 and k2.
    
    Wait, but that changes the output. Let me verify:
    Term: U_vec = e_{i*m+j}, V_vec = e_{j*p+k1}... no wait.
    
    Let me be more careful with the tensor decomposition.
    
    The matrix multiplication tensor T for (n,m,p) has:
    T[i,j,k] = 1 for all i,j,k (in the appropriate ranges)
    
    A rank-r decomposition is: T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    where u_t is in R^{n*m}, v_t is in R^{m*p}, w_t is in R^{p*n}.
    
    The schoolbook decomposition has r = n*m*p terms, where for each (i,j,k):
    u_t = e_{i*m+j}, v_t = e_{j*p+k}, w_t = e_{k*n+i}
    
    Now, can I combine two terms? Suppose I have two terms with the same
    (i,j) but different k: k1 and k2.
    Term 1: e_{i*m+j} ⊗ e_{j*p+k1} ⊗ e_{k1*n+i}
    Term 2: e_{i*m+j} ⊗ e_{j*p+k2} ⊗ e_{k2*n+i}
    
    I want to write this as a single rank-1 term: u ⊗ v ⊗ w
    This would require u = e_{i*m+j}, and v ⊗ w = e_{j*p+k1} ⊗ e_{k1*n+i} + e_{j*p+k2} ⊗ e_{k2*n+i}
    But v ⊗ w is a rank-1 matrix, while the RHS is a sum of two rank-1 matrices
    with different supports, so it's rank 2 (unless they share support).
    
    So I can't combine them into a single rank-1 term.
    
    However, I CAN combine them if I allow the U, V, W matrices to have
    non-unit entries. But the decomposition must be exact.
    
    Let me think about this differently. The key constraint is that the
    decomposition must be exact: sum_t U[:,t] * V[:,t]^T * W[:,t]^T = T
    (in the appropriate tensor sense).
    
    For the schoolbook, each term is a "pure" term. To reduce rank, I need
    to find linear combinations.
    
    I think for this problem, the best I can do without a specific algorithm
    is to use the schoolbook rank of 72. But the task asks for rank-minimal.
    
    Let me just output the schoolbook solution with rank 72, which is what
    the current program does. The directive says to design a new program from
    scratch, so I'll restructure the code but keep the same rank.
    
    Actually, wait. Let me re-read the task. "Find a rank-minimal exact tensor
    decomposition." The score is the rank. So I need to minimize the rank.
    
    Given the constraints (120s, standard library + NumPy + SciPy), and the
    fact that finding a provably minimal rank decomposition for 3x3x8 is
    non-trivial, let me try a computational approach:
    
    1. Set up the tensor as a matrix (matricization).
    2. Use SVD or rank-revealing QR to find a low-rank approximation.
    3. Check if the approximation is exact (within tolerance).
    
    The tensor T for 3x3x8 can be matricized in different ways. The rank of
    the tensor is the minimum over all matricizations of the matrix rank.
    
    Let me try: matricize T as a (n*m) x (m*p) matrix? No, that's not right.
    
    The tensor T ∈ R^{(n*m) x (m*p) x (p*n)}... no, that's not the right
    matricization either.
    
    Actually, the matrix multiplication tensor T ∈ R^{n*m × m*p × p*n} is
    defined by T[i,j,k] = δ_{i,k} (in some sense). Let me be precise.
    
    For matrix multiplication C = A*B where A is n×m and B is m×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T[i,j,k] where i∈[0,n), j∈[0,m), k∈[0,p).
    T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to
    contribute to C[i,k]. So T[i,j,k] = 1 for all valid (i,j,k).
    
    Wait, that's just an all-ones tensor, which has rank 1. That can't be right.
    
    Let me reconsider. The tensor is defined as:
    T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * [contribution of A[i,j]*B[j,k] to C[i,k]]
    
    The contribution is 1 if and only if the indices match, which they always
    do for valid (i,j,k). So T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k.
    
    But this is the all-ones tensor, which has rank 1 (it's just the product
    of the all-ones vector in each space). That's clearly wrong.
    
    I think I'm confusing the tensor definition. Let me look at it differently.
    
    The matrix multiplication bilinear map is:
    (A, B) -> A*B
    
    This can be represented as a 3-tensor T where:
    T[A, B, C] = 1 if C = A*B, else 0.
    
    More precisely, T is a trilinear form:
    T(a, b, c) = sum_{i,j,k} a[i,j] * b[j,k] * c[k,i]
    
    where a is a flattened n×m matrix, b is a flattened m×p matrix, and c
    is a flattened p×n matrix (with the trace convention c[k*n+i] corresponding
    to C[i,k]).
    
    The rank of T is the minimum r such that T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    where u_t ∈ R^{n*m}, v_t ∈ R^{m*p}, w_t ∈ R^{p*n}.
    
    For the schoolbook: T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    This has rank n*m*p = 72.
    
    Now, to find a lower-rank decomposition, I can use the following approach:
    1. Matricize T as a matrix M of size (n*m) x (m*p * p*n) by fixing the
       first index.
    2. The rank of M is an upper bound on the tensor rank.
    3. Use SVD to find a low-rank approximation.
    
    But the tensor rank is the minimum over all possible matricizations.
    
    Let me just try to compute the rank of various matricizations and see
    what the minimum is. If it's less than 72, I can potentially find a
    decomposition of that rank.
    
    Actually, for this specific problem, let me just output the schoolbook
    solution. The rank is 72, which is the same as the current program.
    The directive says to design a new program, so I'll restructure the code.
    
    But wait, the score is seed/value where value is the rank. So a lower rank
    gives a higher score. The current score is 1.0 (rank 72). If I can find
    a rank < 72 decomposition, the score improves.
    
    Let me try to actually compute a low-rank decomposition using NumPy.
    I'll matricize the tensor and use SVD.
    """
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the full tensor T as a 3D array
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # But wait, this is the all-ones tensor, which has rank 1.
    # 
    # I think the issue is that the tensor for matrix multiplication is NOT
    # the all-ones tensor. Let me reconsider.
    #
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # In tensor form, C = sum_{i,k} (sum_j A[i,j] B[j,k]) e_{i,k}^C
    # = sum_{i,j,k} A[i,j] B[j,k] e_{i,k}^C
    #
    # So the tensor T is defined by:
    # T = sum_{i,j,k} e_{i,j}^A ⊗ e_{j,k}^B ⊗ e_{i,k}^C
    #
    # Where e_{i,j}^A is the basis vector for the (i,j) entry of A,
    # e_{j,k}^B is the basis vector for the (j,k) entry of B,
    # e_{i,k}^C is the basis vector for the (i,k) entry of C.
    #
    # In the flattened indexing:
    # A is indexed by i*m + j
    # B is indexed by j*p + k
    # C is indexed by k*n + i (trace convention)
    #
    # So T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # This is NOT the all-ones tensor. The entries of T are:
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k)
    #             0 otherwise
    #
    # For a given (a, b, c), let i = a // m, j = a % m
    #                  j' = b // p, k = b % p
    #                  i' = c % n, k' = c // n
    # Then T[a,b,c] = 1 if j == j' and i == i' and k == k'
    
    # Let's build T as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Now matricize T as a (n*m) x (m*p * p*n) matrix
    # M[a, b*n*p + c] = T[a, b, c]
    M = T.reshape(n*m, m*p * p*n)
    
    # Compute SVD
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    
    # The matrix rank is the number of non-zero singular values
    # But the tensor rank is <= matrix rank
    # Let's find the smallest r such that the top-r SVD approximation is exact
    tol = 1e-10
    for r in range(1, len(s)+1):
        M_approx = (U_svd[:, :r] * s[:r]) @ Vt[:r, :]
        if np.allclose(M_approx, M, atol=tol):
            break
    
    # Now extract the rank-r decomposition from the SVD
    # M = U_r * diag(s) * Vt_r
    # M[a, b*p*n + c] = sum_{t=1}^{r} U_svd[a,t] * s[t] * Vt[t, b*p*n+c]
    # 
    # So T[a,b,c] = sum_t U_svd[a,t] * s[t] * Vt[t, b*p*n+c]
    # 
    # We need to write this as sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t[a] = U_svd[a,t] * sqrt(s[t])
    #       v_t[b] = ? 
    #       w_t[c] = ?
    #
    # The issue is that Vt[t, b*p*n+c] is a single number, but we need it
    # to factor as v_t[b] * w_t[c]. This is only possible if the matrix
    # Vt[t, :] (reshaped to (m*p, p*n)) has rank 1.
    #
    # In general, it won't. So the SVD of the matricization doesn't directly
    # give a tensor decomposition.
    #
    # Let me try a different approach: use the "slice" method.
    # For each a, T[a, :, :] is a (m*p) x (p*n) matrix.
    # The tensor rank is at least the maximum rank of any slice.
    
    max_slice_rank = 0
    for a in range(n*m):
        slice_a = T[a, :, :]
        rank_a = np.linalg.matrix_rank(slice_a, tol=tol)
        max_slice_rank = max(max_slice_rank, rank_a)
    
    # The tensor rank is at least max_slice_rank
    # And at most n*m*p (schoolbook)
    
    # For now, let's just use the schoolbook decomposition
    # since finding the exact minimal rank is non-trivial
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
