"""
Matrix multiplication 2x5 x 5x5: rank-minimal exact tensor decomposition.

Strategy: Use structured Strassen-based decomposition.
Partition the 5x5 multiplication into blocks and apply known low-rank
sub-multiplications (Strassen rank-7 for 2x2x2, schoolbook for remaining).

The key insight: for C = A @ B where A is 2x5, B is 5x5:
- Split columns of A and rows of B into blocks of size 2 and 3.
- A = [A1 (2x2) | A2 (2x3)]
- B = [B1 (5x2); B2 (5x3)]  (split rows into 2+3)
- C = A1@B1 + A2@B2

For A1@B1: 2x2 * 2x2 -> use Strassen (rank 7)
For A2@B2: 2x3 * 3x3 -> decompose further

Actually, let's use a cleaner partition:
- Split the 5 into 2+3
- A (2x5) = [A_2 (2x2), A_3 (2x3)]
- B (5x5) = [[B_22 (2x2), B_23 (2x3)], [B_32 (3x2), B_33 (3x3)]]
- C (2x5) = [C_2, C_3] where:
  C_2 = A_2@B_22 + A_3@B_32
  C_3 = A_2@B_23 + A_3@B_33

For each product:
- A_2 (2x2) @ B_22 (2x2): Strassen rank 7
- A_2 (2x2) @ B_23 (2x3): 2x2*2x3, can be done in rank 4 (schoolbook) or less
- A_3 (2x3) @ B_32 (3x2): 2x3*3x2, rank 6 (schoolbook)
- A_3 (2x3) @ B_33 (3x3): 2x3*3x3, rank 9 (schoolbook)

Total: 7 + 4 + 6 + 9 = 26

But we can do better. Let's use:
- 2x2 * 2x2: Strassen rank 7
- 2x2 * 2x3: We can compute this as 2x2*2x2 + 2x2*2x1 = 7 + 2 = 9? No.
  Actually 2x2 * 2x3: treat as computing C = A*B where A is 2x2, B is 2x3.
  C is 2x3. Schoolbook: 2*2*3 = 12. 
  We can use: C[:,0:2] via Strassen (7) + C[:,2] via 2x2*2x1 (rank 2) = 9.
  Or: C[:,0:2] schoolbook (8) + C[:,2] (2) = 10.
  Better: use Strassen for the 2x2 part (7) + 1 column (2x2*2x1, rank 2) = 9.

- 2x3 * 3x2: C is 2x2. Schoolbook: 2*3*2 = 12.
  We can write this as: sum over 3 middle indices.
  C[i,k] = sum_j A[i,j]*B[j,k]
  This is equivalent to multiplying a 2x3 by 3x2.
  We can use: create 3 rank-1 terms for each (i,k) pair, but share.
  Actually, rank of 2x3*3x2 multiplication tensor: 
  This is the same as (2x3)@(3x2) -> 2x2.
  We can decompose: let A = [a1, a2, a3] (columns), B = [b1, b2, b3] (rows).
  C = a1*b1 + a2*b2 + a3*b3 (outer products summed).
  Each outer product of 2x1 and 1x2 is a rank-1 contribution to the 2x2 result.
  So we need 3 "terms" but each term produces a 2x2 outer product.
  In tensor form: for each middle index j, we have U contributions for A[:,j],
  V contributions for B[j,:], and W contributions for all C positions.
  Rank = 3 * (number of independent output positions that each term touches).
  Actually each term j contributes to ALL 4 entries of C. So rank = 3 * 4 = 12?
  No, that's not right either. Let me think more carefully.
  
  The tensor for 2x3 * 3x2 -> 2x2 has shape (6, 6, 4).
  T[i,j,k] = 1 for all i in 0..1, j in 0..2, k in 0..1.
  That's 6*6*4 = 144 entries, all 1.
  Schoolbook rank = 12 (one for each (i,j,k) triple).
  
  Can we do better? The tensor is all-ones in its "active" subspace.
  Actually it's the all-ones tensor of shape (2,3,2) embedded in (6,6,4).
  The rank of the all-ones tensor of shape (a,b,c) is max(a,b,c) if we can
  factor it, but here the indices are coupled differently.
  
  Let me just use schoolbook for the sub-problems and Strassen where applicable.

Let me use a simpler and more reliable approach:
- Partition 5 = 2 + 3
- Use Strassen (rank 7) for 2x2*2x2 sub-problems
- Use schoolbook for everything else

Sub-problems for C = A(2x5) @ B(5x5):
Split A into A1(2x2), A2(2x3)
Split B into B1(2x5), B2(3x5) [row split]
C = A1@B1 + A2@B2

A1@B1: 2x2 * 2x5 -> 2x5
  = A1 @ [B1a(2x2), B1b(2x3)]
  = [A1@B1a, A1@B1b]
  A1@B1a: 2x2*2x2, Strassen rank 7
  A1@B1b: 2x2*2x3, schoolbook rank 12 (or we can do 2x2*2x2 + 2x2*2x1 = 7+2=9 with Strassen)
  
A2@B2: 2x3 * 3x5 -> 2x5
  = A2 @ [B2a(3x2), B2b(3x3)]
  = [A2@B2a, A2@B2b]
  A2@B2a: 2x3*3x2, schoolbook rank 12
  A2@B2b: 2x3*3x3, schoolbook rank 18

Total with Strassen on 2x2*2x2: 7 + 9 + 12 + 18 = 46
Total all schoolbook: 8 + 12 + 12 + 18 = 50

So Strassen gives rank 46. Let me implement this.
"""

import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def validate(U, V, W, n, m, p):
    """Validate that the decomposition reconstructs the matrix multiplication tensor exactly."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    r = W.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = 0.0
                for c in range(r):
                    val += U[idx_u, c] * V[idx_v, c] * W[idx_w, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def strassen_2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    Returns (U, V, W) where U is (4,7), V is (4,7), W is (4,7).
    
    Strassen's 7 products:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U indices: i*m+j for i in 0..1, j in 0..1 -> 0,1,2,3
    # V indices: j*p+k for j in 0..1, k in 0..1 -> 0,1,2,3
    # W indices: k*n+i for k in 0..1, i in 0..1 -> 0,1,2,3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00(idx0)+A11(idx3), V: B00(idx0)+B11(idx3)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # W: C00(idx0)+C11(idx3)
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    # U: A10(idx2)+A11(idx3), V: B00(idx0)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # W: C10(idx2)+C11(idx3)
    W[2, 1] = 1; W[3, 1] = -1  # C11 = M1 - M2 + ...
    
    # M3 = A00*(B01-B11)
    # U: A00(idx0), V: B01(idx1)-B11(idx3)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # W: C01(idx1)+C11(idx3)
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    # U: A11(idx3), V: B10(idx2)-B00(idx0)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # W: C00(idx0)+C10(idx2)
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    # U: A00(idx0)+A01(idx1), V: B11(idx3)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # W: -C00(idx0)+C01(idx1)
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10(idx2)-A00(idx0), V: B00(idx0)+B01(idx1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # W: C11(idx3)
    W[3, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01(idx1)-A11(idx3), V: B10(idx2)+B11(idx3)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # W: C00(idx0)
    W[0, 6] = 1
    
    return U, V, W


def schoolbook_block(n1, m1, p1, row_offset_u, row_offset_v, row_offset_w, col_offset):
    """
    Return schoolbook decomposition for n1 x m1 * m1 x p1 -> n1 x p1.
    The indices are offset to place in the larger matrices.
    Returns (U, V, W) of shapes (n1*m1, r), (m1*p1, r), (p1*n1, r)
    where r = n1*m1*p1, and a list of (local_idx_u, local_idx_v, local_idx_w, col) mappings.
    """
    r = n1 * m1 * p1
    U = np.zeros((n1 * m1, r))
    V = np.zeros((m1 * p1, r))
    W = np.zeros((p1 * n1, r))
    col = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U[i * m1 + j, col] = 1
                V[j * p1 + k, col] = 1
                W[k * n1 + i, col] = 1
                col += 1
    return U, V, W


def strassen_2x2_in_context(n1, m1, p1):
    """
    Get Strassen decomposition for a 2x2*2x2 subproblem embedded in larger context.
    Returns U(4,7), V(4,7), W(4,7) for the local 2x2*2x2 -> 2x2 problem.
    """
    return strassen_2x2()


def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    start_time = time.time()
    deadline = 140
    
    # Partition: split the middle dimension (5) into 2+3
    # A is 2x5: A = [A1 (2x2), A2 (2x3)]
    # B is 5x5: B = [[B11 (2x2), B12 (2x3)], [B21 (3x2), B22 (3x3)]]
    # C = A@B = [A1@B11 + A2@B21, A1@B12 + A2@B22]
    # C is 2x5: C = [C1 (2x2), C2 (2x3)]
    
    # Sub-problems:
    # P1: A1(2x2) @ B11(2x2) -> 2x2, use Strassen rank 7
    # P2: A1(2x2) @ B12(2x3) -> 2x3, schoolbook rank 12
    # P3: A2(2x3) @ B21(3x2) -> 2x2, schoolbook rank 12
    # P4: A2(2x3) @ B22(3x3) -> 2x3, schoolbook rank 18
    # Total: 7 + 12 + 12 + 18 = 49
    
    # Wait, let me reconsider. For P2: 2x2 * 2x3 -> 2x3
    # Schoolbook: 2*2*3 = 12
    # Can we do better? Split B12 into B12a(2x2) and B12b(2x1):
    #   A1@B12a: 2x2*2x2, Strassen rank 7
    #   A1@B12b: 2x2*2x1, schoolbook rank 4
    #   Total: 7 + 4 = 11
    # Or: A1@B12a: 2x2*2x2 schoolbook rank 8, A1@B12b: 2x2*2x1 rank 4, total 12.
    # Better: Strassen for 2x2 part: 7 + 4 = 11.
    
    # For P3: 2x3 * 3x2 -> 2x2
    # Schoolbook: 2*3*2 = 12
    # Can we do better? This is the "dot product" type.
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    # We can write: for each j, the contribution is outer(A[:,j], B[j,:])
    # In tensor form, each j gives a rank-3 term (2*1*2 = 4 entries but structured)
    # Actually each j contributes to all 4 entries of C, so it's rank 4 per j? No.
    # The tensor entry T[i,j,k] = 1 for all i,j,k.
    # Rank-1 decomposition: we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # For fixed j, U[i,j,c] and V[j,k,c] and W[k,i,c] must multiply to give 1 for all i,k.
    # So for each j, we need U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for all i,k.
    # That's 1 term per j, giving rank 3. But wait, each term has U nonzero for 2 rows,
    # V nonzero for 2 rows, W nonzero for 4 rows.
    # Actually: for term c corresponding to middle index j:
    #   U[i*m+j, c] = 1 for i=0,1
    #   V[j*p+k, c] = 1 for k=0,1
    #   W[k*n+i, c] = 1 for k=0,1, i=0,1
    # This works! Rank = 3 for 2x3*3x2.
    
    # Let me verify: sum_c U[i*3+j, c]*V[j*2+k, c]*W[k*2+i, c]
    # For the 3 terms (j=0,1,2):
    #   term j: U[i*3+j, j]=1, V[j*2+k, j]=1, W[k*2+i, j]=1
    #   Product = 1 for each term
    #   Sum over j = 3, not 1!
    
    # Oops. The issue is that each term contributes 1, and we sum 3 terms to get 3, not 1.
    # We need the sum to be 1, not 3. So we need to divide by 3? But entries must be integer/half-integer.
    # 1/3 is not integer or half-integer.
    
    # Alternative: use the fact that we can have different coefficients.
    # We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k.
    # If we use 3 terms (one per middle index), each giving product 1, sum = 3.
    # We need sum = 1. So we can't simply use 3 identical terms.
    
    # What if we use a different decomposition?
    # For 2x3*3x2, the tensor is T[i,j,k]=1 for i in 0,1; j in 0,1,2; k in 0,1.
    # This is a 2x3x2 all-ones tensor.
    # The rank of the all-ones tensor of shape (a,b,c) is... 
    # For the all-ones tensor J_{a,b,c}, we can write it as:
    #   J = sum_{l=1}^{min(a,b,c)} u_l * v_l * w_l
    # where u_l, v_l, w_l are all-ones vectors? No, that gives rank min(a,b,c) only
    # if the vectors are all-ones, but then u_l*v_l*w_l is the all-ones tensor of shape (a,b,c)
    # only for l=1. For l>1, we'd need different vectors.
    
    # Actually, the all-ones tensor of shape (a,b,c) has rank min(a,b,c) if we allow
    # real entries, by using:
    #   J[i,j,k] = sum_{l=1}^{r} u_l[i] * v_l[j] * w_l[k]
    # where r = min(a,b,c) and we use Fourier-like vectors.
    # But for integer/half-integer entries, this is harder.
    
    # For 2x3x2 all-ones: rank with integer entries?
    # We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k.
    # With r=3 (one per middle index), we get sum=3. 
    # With r=6, we could use: for each (i,k) pair, have terms that cancel.
    # This is getting complicated. Let me just use schoolbook rank 12 for P3.
    
    # Revised plan:
    # P1: 2x2*2x2, Strassen rank 7
    # P2: 2x2*2x3, split into 2x2*2x2 (Strassen 7) + 2x2*2x1 (schoolbook 4) = 11
    # P3: 2x3*3x2, schoolbook rank 12
    # P4: 2x3*3x3, schoolbook rank 18
    # Total: 7 + 11 + 12 + 18 = 48
    
    # Can we do better on P4? 2x3*3x3 -> 2x3
    # Split into 2x3*3x2 + 2x3*3x1:
    #   2x3*3x2: schoolbook 12 (or better?)
    #   2x3*3x1: schoolbook 6
    #   Total: 18
    # Or split differently: 2x2*3x3 + 2x1*3x3:
    #   2x2*3x3: schoolbook 12
    #   2x1*3x3: schoolbook 6
    #   Total: 18
    # No improvement.
    
    # What about P2: 2x2*2x3 -> 2x3
    # Alternative: treat as 2x2 * [2x2 | 2x1]
    #   2x2*2x2: Strassen 7
    #   2x2*2x1: schoolbook 4
    #   Total: 11
    # Or: 2x2*2x3 directly. The tensor is T[i,j,k]=1 for i in 0,1; j in 0,1; k in 0,1,2.
    # Shape (2,2,3) all-ones. 
    # With schoolbook: 12 terms.
    # With Strassen on the 2x2 part + 1 extra column: 7 + 4 = 11.
    
    # Let me also reconsider P3: 2x3*3x2 -> 2x2
    # The tensor has shape (6,6,4) with T[i,j,k]=1 for all valid i,j,k.
    # Actually it's the "matmul" tensor for 2x3 * 3x2.
    # Known: the rank of the 2x3*3x2 matmul tensor... 
    # We can write it as: C = A*B where A is 2x3, B is 3x2.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    # This is a sum of 3 rank-1 tensors (outer products of A[:,j] and B[j,:]).
    # In the tensor decomposition framework:
    #   For each j=0,1,2: U[i*3+j, c_j] = A[i,j] coefficient = 1
    #                      V[j*2+k, c_j] = B[j,k] coefficient = 1  
    #                      W[k*2+i, c_j] = C[k,i] coefficient = 1
    #   Sum over c: for each (i,j,k), sum_c U[i*3+j,c]*V[j*2+k,c]*W[k*2+i,c]
    #   Only term c=j contributes: 1*1*1 = 1. 
    #   Wait, that's correct! Each (i,j,k) triple has exactly one term (c=j) that
    #   gives nonzero product, and it equals 1.
    #   So rank = 3!
    
    # Let me double-check: 
    #   U has shape (6, 3), V has shape (6, 3), W has shape (4, 3)
    #   U[i*3+j, j] = 1 for all i, j (and 0 otherwise)
    #   V[j*2+k, j] = 1 for all j, k (and 0 otherwise)
    #   W[k*2+i, j] = 1 for all k, i, j (and 0 otherwise)
    #   
    #   For a given (i, j, k):
    #     sum_c U[i*3+j, c] * V[j*2+k, c] * W[k*2+i, c]
    #     = U[i*3+j, j] * V[j*2+k, j] * W[k*2+i, j]  (only c=j is nonzero in U and V)
    #     = 1 * 1 * 1 = 1  ✓
    
    # Great! So P3 has rank 3, not 12!
    
    # Similarly, let's re-examine P4: 2x3*3x3 -> 2x3
    # C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    # For each j: U[i*3+j, j] = 1, V[j*3+k, j] = 1, W[k*2+i, j] = 1
    # Rank = 3? Let's verify:
    #   U shape (6, 3), V shape (9, 3), W shape (6, 3)
    #   For (i, j, k): sum_c U[i*3+j,c]*V[j*3+k,c]*W[k*2+i,c]
    #   = U[i*3+j,j]*V[j*3+k,j]*W[k*2+i,j] = 1*1*1 = 1 ✓
    #   Rank = 3!
    
    # And P2: 2x2*2x3 -> 2x3
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k]
    # For each j: U[i*2+j, j] = 1, V[j*3+k, j] = 1, W[k*2+i, j] = 1
    # Rank = 2!
    # Verify: U shape (4,2), V shape (6,2), W shape (6,2)
    #   For (i,j,k): sum_c U[i*2+j,c]*V[j*3+k,c]*W[k*2+i,c]
    #   = U[i*2+j,j]*V[j*3+k,j]*W[k*2+i,j] = 1 ✓
    #   Rank = 2!
    
    # And P1: 2x2*2x2 -> 2x2
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k]
    # For each j: U[i*2+j, j] = 1, V[j*2+k, j] = 1, W[k*2+i, j] = 1
    # Rank = 2!
    # Verify: U shape (4,2), V shape (4,2), W shape (4,2)
    #   For (i,j,k): sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c]
    #   = U[i*2+j,j]*V[j*2+k,j]*W[k*2+i,j] = 1 ✓
    #   Rank = 2!
    
    # Wait, this is just the schoolbook algorithm expressed differently!
    # The "rank" here is the number of middle indices, which is the standard
    # schoolbook rank. For 2x2*2x2, that's 2, not 8.
    
    # I think I'm confusing two different things. Let me re-read the contract.
    
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k.
    # A rank-1 term is U_vec * V_vec * W_vec where:
    #   U_vec is in R^{n*m}, V_vec is in R^{m*p}, W_vec is in R^{p*n}
    #   The product (outer) gives a tensor with entry [i*m+j, j*p+k, k*n+i] 
    #   = U[i*m+j] * V[j*p+k] * W[k*n+i]
    #   
    #   So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # For the "for each middle index j" approach:
    #   Term c=j: U[i*m+j, j] = 1 for all i, V[j*p+k, j] = 1 for all k, W[k*n+i, j] = 1 for all i,k
    #   Then T[i,j',k] = sum_c U[i*m+j', c] * V[j'*p+k, c] * W[k*n+i, c]
    #   For j' = j: only c=j contributes, giving 1*1*1 = 1 ✓
    #   For j' ≠ j: U[i*m+j', j] = 0, so the term is 0.
    #   But we need T[i,j',k] = 1 for ALL j', not just j'=j!
    
    # I see the error. The tensor entry T[i,j',k] must equal 1 for ALL j', not just
    # when j'=j. My "for each j" decomposition only gives 1 when j'=j and 0 otherwise.
    # That's not the all-ones tensor!
    
    # The correct schoolbook decomposition:
    #   For each (i, j, k) triple, have a separate term c:
    #   U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    #   This gives T[i,j,k] = 1 (only the term with matching indices contributes)
    #   and T[i,j',k] = 0 for j' ≠ j (no term has U[i*m+j',c]=1 when c is for (i,j,k))
    #   
    #   Wait, that's still not right. For T[i,j',k] with j'≠j:
    #   sum_c U[i*m+j',c]*V[j'*p+k,c]*W[k*n+i,c]
    #   The term c for (i,j,k) has U[i*m+j,c]=1 but U[i*m+j',c]=0 (since j'≠j).
    #   So no term contributes, giving 0. But we need 1!
    
    # I'm fundamentally confused. Let me re-read the problem.
    
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # "exactly decompose the matrix multiplication tensor"
    
    # The matrix multiplication tensor for n x m * m x p -> n x p is:
    #   T[i, j, k] = 1 if the (i,j) element of A and (j,k) element of B 
    #   contribute to (i,k) element of C.
    #   For standard matmul, C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k]=1 for all i,j,k.
    
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Schoolbook: one term per (i,j,k) triple.
    #   Term (i0,j0,k0): U[i0*m+j0, c] = 1, V[j0*p+k0, c] = 1, W[k0*n+i0, c] = 1
    #   For T[i,j,k]: only the term with i0=i, j0=j, k0=k contributes.
    #   So T[i,j,k] = 1 for the matching term, 0 for others. Sum = 1. ✓
    
    # But wait, that means T[i,j,k] = 1 only when there's a term for exactly (i,j,k),
    # and 0 otherwise. Since we have a term for every (i,j,k), T[i,j,k] = 1 for all. ✓
    
    # OK so my earlier "for each j" approach was wrong because it only gave 1 for
    # j'=j and 0 for j'≠j. The correct schoolbook has a term for each (i,j,k).
    
    # Now, can we do better than schoolbook? Yes, via Strassen-like tricks.
    # The key is to use linear combinations: instead of having U[i*m+j,c]=1 for a
    # single (i,j), we can have U[i*m+j,c] = a and U[i'*m+j',c] = b for different
    # (i,j) pairs, as long as the products work out.
    
    # For Strassen on 2x2*2x2:
    #   M1 = (A00+A11)*(B00+B11)
    #   In tensor terms: U[0,c]=1, U[3,c]=1 (A00 and A11), V[0,c]=1, V[3,c]=1 (B00 and B11)
    #   W[0,c]=1, W[3,c]=1 (C00 and C11)
    #   This term contributes 1 to T[0,0,0], T[0,0,1]... wait no.
    
    # Let me be more careful. For 2x2*2x2:
    #   n=2, m=2, p=2
    #   U indices: i*2+j for i,j in 0,1 -> 0,1,2,3
    #   V indices: j*2+k for j,k in 0,1 -> 0,1,2,3
    #   W indices: k*2+i for k,i in 0,1 -> 0,1,2,3
    #   
    #   M1 = (A00+A11)*(B00+B11) contributes to C00 and C11.
    #   Term c: U[0,c]=1 (A00), U[3,c]=1 (A11), V[0,c]=1 (B00), V[3,c]=1 (B11)
    #   W[0,c]=1 (C00), W[3,c]=1 (C11)
    #   
    #   Contribution to T[i,j,k]: U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c]
    #   T[0,0,0]: U[0,c]*V[0,c]*W[0,c] = 1*1*1 = 1
    #   T[0,0,1]: U[0,c]*V[1,c]*W[1,c] = 1*0*0 = 0  (V[1,c]=0 since V[0,c]=1,V[3,c]=1)
    #   T[0,1,0]: U[1,c]*V[0,c]*W[0,c] = 0*1*1 = 0
    #   T[1,0,0]: U[2,c]*V[0,c]*W[0,c] = 0*1*1 = 0
    #   T[1,1,1]: U[3,c]*V[3,c]*W[3,c] = 1*1*1 = 1
    #   
    #   So M1 contributes 1 to T[0,0,0] and T[1,1,1], and 0 to all others.
    #   That's correct for the Strassen formula: M1 adds to C00 and C11.
    
    # OK so the Strassen decomposition works. Let me now build the full 2x5*5x5
    # decomposition using Strassen on the 2x2*2x2 sub-block and schoolbook elsewhere.
    
    # Full problem: n=2, m=5, p=5
    # U: 10 x r, V: 25 x r, W: 10 x r
    # 
    # Partition middle index j (0..4) into j in {0,1} (size 2) and j in {2,3,4} (size 3)
    # Partition output index k (0..4) into k in {0,1} (size 2) and k in {2,3,4} (size 3)
    #
    # C[i,k] = sum_{j=0}^{4} A[i,j]*B[j,k]
    #        = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
    #
    # For k in {0,1}:
    #   C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
    #   First sum: 2x2*2x2 (A[:,0:2] @ B[0:2,0:2]) -> use Strassen rank 7
    #   Second sum: 2x3*3x2 (A[:,2:5] @ B[2:5,0:2]) -> schoolbook rank 12
    #
    # For k in {2,3,4}:
    #   C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{4} A[i,j]*B[j,k]
    #   First sum: 2x2*2x3 (A[:,0:2] @ B[0:2,2:5]) -> schoolbook rank 12
    #   Second sum: 2x3*3x3 (A[:,2:5] @ B[2:5,2:5]) -> schoolbook rank 18
    #
    # Total: 7 + 12 + 12 + 18 = 49
    
    # Can we improve? Let's try Strassen on more sub-blocks.
    # For 2x2*2x3: split B into 2x2 + 2x1:
    #   2x2*2x2: Strassen 7
    #   2x2*2x1: schoolbook 4
    #   Total: 11 (saves 1)
    #
    # For 2x3*3x2: This is trickier. 
    #   A is 2x3, B is 3x2. C is 2x2.
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #   We can write this as: 
    #     Let A' = [A[:,0:2]] (2x2), A'' = A[:,2] (2x1)
    #     Let B' = B[0:2,:] (2x2), B'' = B[2,:] (1x2)
    #     C = A'@B' + A''@B''
    #     A'@B': 2x2*2x2, Strassen 7
    #     A''@B'': 2x1*1x2, this is an outer product. Rank 4 (one for each C entry) or...
    #       C[i,k] = A''[i]*B''[k]
    #       In tensor form: U[i*3+2, c] = A''[i] coeff = 1, V[2*2+k, c] = B''[k] coeff = 1, W[k*2+i, c] = 1
    #       For (i, j, k): only j=2 contributes (since U[i*3+2,c]=1 only for j=2)
    #       T[i,2,k] = U[i*3+2,c]*V[2*2+k,c]*W[k*2+i,c] = 1*1*1 = 1
    #       T[i,j,k] for j≠2: U[i*3+j,c] = 0, so 0.
    #       But we need T[i,j,k]=1 for ALL j, not just j=2!
    #       
    #       Same issue as before. The "for each j" approach doesn't work because
    #       the tensor requires T[i,j,k]=1 for all j.
    
    # I think the fundamental issue is that for the full tensor T[i,j,k]=1 for all i,j,k,
    # we need the sum over c to give 1 for every (i,j,k) triple. The Strassen trick works
    # by having linear combinations that "spread" the contribution across multiple (i,j,k)
    # entries in a coordinated way.
    
    # Let me just implement the 49-rank solution (Strassen on one 2x2*2x2 block)
    # and verify it works.
    
    total_rank = 49
    U = np.zeros((10, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((10, total_rank))
    col = 0
    
    # P1: A[:,0:2] @ B[0:2,0:2] -> C[:,0:2] (2x2*2x2, Strassen rank 7)
    # Local indices: i in 0,1; j in 0,1; k in 0,1
    # U local: i*2+j -> 0,1,2,3; global U: i*5+j
    # V local: j*2+k -> 0,1,2,3; global V: j*5+k
    # W local: k*2+i -> 0,1,2,3; global W: k*2+i
    
    sU, sV, sW = strassen_2x2()
    for c in range(7):
        # Map local U indices to global
        for i in range(2):
            for j in range(2):
                U[i*5+j, col+c] = sU[i*2+j, c]
        for j in range(2):
            for k in range(2):
                V[j*5+k, col+c] = sV[j*2+k, c]
        for k in range(2):
            for i in range(2):
                W[k*2+i, col+c] = sW[k*2+i, c]
    col += 7  # col = 7
    
    # P2: A[:,0:2] @ B[0:2,2:5] -> C[:,2:5] (2x2*2x3, schoolbook rank 12)
    # i in 0,1; j in 0,1; k in 2,3,4
    # U global: i*5+j; V global: j*5+k; W global: k*2+i
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    # col = 7 + 12 = 19
    
    # P3: A[:,2:5] @ B[2:5,0:2] -> C[:,0:2] (2x3*3x2, schoolbook rank 12)
    # i in 0,1; j in 2,3,4; k in 0,1
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    # col = 19 + 12 = 31
    
    # P4: A[:,2:5] @ B[2:5,2:5] -> C[:,2:5] (2x3*3x3, schoolbook rank 18)
    # i in 0,1; j in 2,3,4; k in 2,3,4
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i*5+j, col] = 1
                V[j*5+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    # col = 31 + 18 = 49
    
    assert col == total_rank, f"Expected {total_rank}, got {col}"
    
    # Validate
    if not validate(U, V, W, n, m, p):
        # Fallback to schoolbook
        U = np.zeros((10, 50))
        V = np.zeros((25, 50))
        W = np.zeros((10, 50))
        col = 0
        for i in range(2):
            for j in range(5):
                for k in range(5):
                    U[i*5+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
    
    # Convert to list of lists with integer values
    U_list = [[int(U[i][c]) if U[i][c] == int(U[i][c]) else float(U[i][c]) for c in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[int(V[i][c]) if V[i][c] == int(V[i][c]) else float(V[i][c]) for c in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[int(W[i][c]) if W[i][c] == int(W[i][c]) else float(W[i][c]) for c in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
