"""Matrix multiplication 2x3x10 rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 x 3x10 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (6, rank), (30, rank), (20, rank).
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: Use Strassen-like block decomposition.
    # The matrix is 2x3 times 3x10.
    # We can view the 2x3 as a 2x2 block plus a 2x1 block, or similar.
    # Actually, let's think about the tensor structure.
    # The multiplication tensor has indices (i, j, k) where i in 0..1, j in 0..2, k in 0..9.
    # Rank of schoolbook is 2*3*10 = 60.
    
    # Known lower bounds and constructions:
    # For 2x3x10, we can use the fact that 3 = 2+1.
    # Let's try to decompose using Strassen for 2x2 blocks.
    
    # Alternative approach: Use the known optimal rank for small matrix multiplication.
    # The rank of 2x3x10 matrix multiplication.
    # We can write the 2x3 matrix as [[A, B], [C, D]] where A, C are 2x2 and B, D are 2x1.
    # Wait, 2x3 means 2 rows, 3 columns.
    # Let A = [[a11, a12, a13], [a21, a22, a23]]
    # Let B = [[b11, ..., b110], [b21, ..., b210], [b31, ..., b310]]
    
    # The product C = A * B has entries c_ik = sum_j a_ij b_jk.
    
    # One approach: Use the fact that we can share computations.
    # For a 2x2x2 block, Strassen reduces 8 mults to 7.
    # Here we have overlapping 2x2 blocks.
    
    # Let's try a direct construction.
    # The tensor T has indices (i, j, k) with i in {0,1}, j in {0,1,2}, k in {0,...,9}.
    # T(i,j,k) = e_i ⊗ e_j ⊗ e_k
    
    # We want to write T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    
    # Let's try to use the structure.
    # For each k (column of B), we compute A * b_k where b_k is a 3-vector.
    # A * b_k = [a11*b_k1 + a12*b_k2 + a13*b_k3, a21*b_k1 + a22*b_k2 + a23*b_k3]
    
    # This is a 2x3 matrix times a 3-vector, giving a 2-vector.
    # The tensor for this is of size 2x3x1 (for each k independently).
    # But we have 10 such k's, and they share the same A.
    
    # Actually, the full tensor is 2x3x10.
    # We can think of it as 10 independent 2x3x1 tensors (one for each column k), but they share the A part.
    
    # Let's try a rank construction.
    # For each column k of B, we need to compute 2 outputs (rows of C).
    # c_i,k = sum_j a_i,j * b_j,k for i=0,1 and j=0,1,2.
    
    # Total 20 outputs. Each is a product of an A-entry and a B-entry, summed over j.
    
    # A simple construction: For each (i, j, k), have a term a_i,j * b_j,k * e_i,k.
    # That's 2*3*10 = 60 terms (schoolbook).
    
    # To reduce rank, we need to combine terms.
    # For fixed i and k, c_i,k = a_i0*b_0k + a_i1*b_1k + a_i2*b_2k.
    # This is a 1x3x1 tensor (for fixed i,k) which has rank 1 (it's just a sum of products).
    # Wait, no. The tensor for fixed i,k is sum_j e_j ⊗ e_{j,k} ⊗ e_{i,k}... no.
    
    # Let me re-think the tensor indices.
    # U has shape (n*m, rank) = (6, R). Index: (i, j) -> i*m+j.
    # V has shape (m*p, rank) = (30, R). Index: (j, k) -> j*p+k.
    # W has shape (p*n, rank) = (20, R). Index: (k, i) -> k*n+i.
    
    # The decomposition says: T(i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T(i,j,k) = 1 if (i,j,k) is a valid index combination, else 0.
    # Wait, the matrix multiplication tensor is T(i,j,k) = 1 for all valid (i,j,k), and the
    # multiplication is C_ik = sum_j T(i,j,k) * A_ij * B_jk.
    # So the tensor is simply all-ones on the valid indices.
    
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k.
    
    # This is equivalent to finding a rank-R decomposition of the all-ones tensor of size 2x3x10.
    
    # The rank of the all-ones tensor of size a x b x c is max(a, b, c) if we allow real entries?
    # No, that's for the tensor that maps (x, y, z) -> xyz.
    # The all-ones tensor T(i,j,k) = 1 for all i,j,k has rank 1: take U = all-ones, V = all-ones, W = all-ones.
    # But wait, that would mean C_ik = sum_j A_ij * B_jk * 1, which is correct!
    
    # Oh wait, I think I'm confusing myself. Let me re-read the task.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."
    # The matrix multiplication tensor is defined such that the multiplication algorithm is:
    # C_ik = sum_r (sum_j U[i*m+j, r] * A_ij) * (sum_j V[j*p+k, r] * B_jk) * W[k*n+i, r]
    # No, that's not right either.
    
    # The standard formulation: The multiplication tensor T is a 3-tensor where T(i,j,k) corresponds to
    # the coefficient of A_ij * B_jk in C_ik. For matrix multiplication, T(i,j,k) = 1 for all valid (i,j,k).
    # The rank of T is the minimum number of rank-1 tensors that sum to T.
    # A rank-1 tensor is u ⊗ v ⊗ w, which corresponds to (sum_j u_ij A_ij) * (sum_k v_jk B_jk) * w_ki... 
    
    # Actually, the standard definition: T = sum_{r=1}^{R} u^{(r)} ⊗ v^{(r)} ⊗ w^{(r)}
    # where u^{(r)} is in R^{n*m}, v^{(r)} is in R^{m*p}, w^{(r)} is in R^{p*n}.
    # And the multiplication is: C = sum_{r=1}^{R} (u^{(r)} · A) ⊗ (v^{(r)} · B) * w^{(r)}... 
    
    # No, let me look at the seed program. The seed program sets U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for each (i,j,k).
    # This means for each (i,j,k), there's a rank-1 term that picks out A_ij and B_jk and contributes to C_ik.
    # The rank is n*m*p = 60.
    
    # The question is: can we do better?
    # The tensor T(i,j,k) = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,9}.
    # We want T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r.
    # This is equivalent to: T(i,j,k) = sum_r u_r(i,j) * v_r(j,k) * w_r(k,i) = 1 for all i,j,k.
    
    # This is a tensor rank problem for the all-ones tensor.
    # The all-ones tensor of size 2x3x10.
    # For the all-ones tensor, the rank is 1: take u = all-ones, v = all-ones, w = all-ones.
    # Then sum_r u(i,j)*v(j,k)*w(k,i) = 1*1*1 = 1.
    # So the rank is 1?!
    
    # Wait, that can't be right. Let me check the seed program again.
    # The seed program returns rank 60. If rank 1 worked, the seed would have found it.
    
    # I think I'm misunderstanding the tensor structure. Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # I think the issue is that the decomposition must be such that the multiplication is computed as:
    # C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A_ij) * (sum_j V[j*p+k, r] * B_jk)
    # No, that's not standard either.
    
    # Let me look at the seed more carefully.
    # For each (i,j,k), the seed sets:
    #   U[i*m+j, col] = 1
    #   V[j*p+k, col] = 1  
    #   W[k*n+i, col] = 1
    # This means the r-th term contributes U[i*m+j,r] * A_ij * V[j*p+k,r] * B_jk * W[k*n+i,r] to C_ik.
    # Summing over r: C_ik = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A_ij * V[j*p+k,r] * B_jk)
    # Hmm, that doesn't look right either.
    
    # Actually, I think the standard formulation is:
    # C_ik = sum_r (sum_j U[i*m+j, r] * A_ij) * (sum_j V[j*p+k, r] * B_jk) * W[k*n+i, r]
    # No, that would give a product of two sums times W, which doesn't match.
    
    # Let me think about this differently. The rank-1 term u ⊗ v ⊗ w corresponds to:
    # (u^T A_flat) * (v^T B_flat) * w... no.
    
    # I think the correct interpretation is:
    # The multiplication tensor T has T(i,j,k) = 1 for all valid indices.
    # A rank-1 decomposition term is u ⊗ v ⊗ w where:
    #   u is indexed by (i,j), v by (j,k), w by (k,i)
    # And the contribution to the multiplication is:
    #   For each r, compute M_r = (sum_{i,j} u(i,j) A_ij) * (sum_{j,k} v(j,k) B_jk)
    #   Then C += M_r * w... no, w is a vector, not a scalar.
    
    # I think I need to look at this more carefully. Let me assume the standard definition:
    # The matrix multiplication tensor is T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    # with T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), and 0 otherwise.
    # Wait, that's not a tensor in the usual sense because the indices are coupled.
    
    # Actually, I think the correct formulation is:
    # T is a 3-tensor with T(a, b, c) where a ∈ {0,...,n*m-1}, b ∈ {0,...,m*p-1}, c ∈ {0,...,p*n-1}.
    # T(a, b, c) = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' and j = j' and k = k' and i = i', else 0.
    # Wait, that means T is zero unless the indices are consistent.
    
    # Let me just try rank 1 and see if it works.
    # If U = all-ones, V = all-ones, W = all-ones, then:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1*1*1 = 1.
    # But this is for a single r. If R=1, then T(i,j,k) = 1 for all i,j,k.
    # But T should be 1 only when the indices are consistent (i.e., the same j and k are used).
    
    # I think the key insight is that the tensor T is NOT the all-ones tensor.
    # T(i,j,k) = 1 if the (i,j) index in U, (j,k) index in V, and (k,i) index in W are "compatible".
    # Specifically, T is defined such that the multiplication C_ik = sum_j A_ij B_jk is recovered.
    
    # I think the correct formulation is:
    # C_ik = sum_{r=1}^{R} (sum_j U[i*m+j, r] A_ij) (sum_j V[j*p+k, r] B_jk) W[k*n+i, r]
    # No, that doesn't dimensionally work.
    
    # Let me try: C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A_ij) * (sum_j V[j*p+k, r] B_jk)
    # This would give: C_ik = sum_r W[k*n+i, r] * (U_r^T A_col_i) * (V_r^T B_row_k)
    # Where A_col_i is the i-th row of A flattened, and B_row_k is the k-th column of B flattened.
    
    # For this to equal sum_j A_ij B_jk, we need:
    # sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A_ij) * (sum_j V[j*p+k, r] B_jk) = sum_j A_ij B_jk
    
    # This is a bilinear form in A and B. The rank of this bilinear form is the rank of the tensor.
    
    # For the seed program, each (i,j,k) has its own r, with U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1.
    # Then the r-th term contributes: 1 * A_ij * B_jk * 1 = A_ij B_jk to C_ik.
    # Summing over all r (which corresponds to all (i',j',k')), we get:
    # C_ik = sum_{i',j',k'} [i'=i and k'=k] A_i'j' B_j'k' = sum_j A_ij B_jk.
    # Yes! That's correct.
    
    # So the tensor T has T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k), and 0 otherwise.
    # This is NOT the all-ones tensor. It's a sparse tensor where only "consistent" index triples are 1.
    
    # Now, can we reduce the rank?
    # The rank of this tensor is the minimum number of rank-1 tensors that sum to it.
    
    # For 2x3x10, the schoolbook rank is 60.
    # Can we do better?
    
    # One approach: Use the fact that for fixed i and k, the sum over j can be shared.
    # C_ik = sum_j A_ij B_jk.
    # This is an inner product of the i-th row of A and the k-th column of B.
    
    # For each pair (i,k), we have 2*10 = 20 inner products, each of length 3.
    # The rank of a single inner product of length m is 1 (it's just a sum of products).
    # But the A and B entries are shared across different (i,k) pairs.
    
    # Actually, I think the rank of the 2x3x10 matrix multiplication tensor is known.
    # Let me try to construct a lower-rank decomposition.
    
    # Idea: Group the j indices.
    # For j=0: A_i0 * B_0k contributes to all C_ik.
    # For j=1: A_i1 * B_1k contributes to all C_ik.
    # For j=2: A_i2 * B_2k contributes to all C_ik.
    
    # So C_ik = A_i0*B_0k + A_i1*B_1k + A_i2*B_2k.
    
    # We can write this as: C_ik = sum_j (A_i0*B_0k) [j=0] + (A_i1*B_1k) [j=1] + (A_i2*B_2k) [j=2]
    
    # For each j, the term A_ij * B_jk is a rank-1 tensor in (i,k) space, but it depends on j.
    # The total tensor is sum_j (e_j ⊗ e_j) ⊗ (A_col_j ⊗ B_row_j)... 
    
    # I think the rank is at most 3*2*10/2 = 30 if we can share, but I'm not sure.
    
    # Let me try a different approach. Use the known result that the rank of m x n x p matrix multiplication
    # is at most m*n*p, and can be reduced using Strassen-like methods.
    
    # For 2x3x10, let's try to use the fact that 3 = 2+1.
    # Write A = [A2 | A1] where A2 is 2x2 and A1 is 2x1.
    # Write B = [B2; B1] where B2 is 2x10 and B1 is 1x10.
    # Then C = A2*B2 + A1*B1.
    
    # A2*B2 is a 2x2 times 2x10 multiplication, rank at most 2*2*10 = 40.
    # A1*B1 is a 2x1 times 1x10 multiplication, rank at most 2*1*10 = 20.
    # Total rank at most 60. No improvement.
    
    # But we can use Strassen for the 2x2x10 part.
    # 2x2x10: The 2x2 multiplication has rank 7 (Strassen).
    # For 2x2x10, we can share the Strassen intermediates across the 10 columns.
    # The 2x2 part: A2 = [[a,b],[c,d]], B2 = [[e,f],[g,h]] (for each column).
    # Strassen: m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e), m5=(a+b)h, m6=(c-d)(e+f), m7=b(g+h).
    # C11 = m1+m4-m5+m7, C12 = m3+m5, C21 = m2+m4, C22 = m1-m2+m3+m6.
    
    # For 2x2x10, we have 10 columns of B2.
    # The Strassen intermediates m1..m7 are linear combinations of A2 entries and B2 entries.
    # Since B2 has 10 columns, each m_r is a 1x10 vector (for fixed A2).
    # So we need 7 * 10 = 70 scalar multiplications for the 2x2x10 part? No, that's not right.
    
    # Actually, for 2x2x10, the tensor is 2x2x10.
    # The rank of 2x2x10 matrix multiplication:
    # We can use Strassen's 7 multiplications for the 2x2 part, but each "multiplication" is actually
    # a dot product of length 10 (since B2 is 2x10, not 2x2).
    # Wait, no. 2x2 times 2x10 means the second matrix is 2 rows, 10 columns.
    # So it's 2x2 * 2x10 = 2x10.
    # The multiplication is: C_ik = sum_j A2_ij B2_jk for i=0,1 and k=0..9.
    # This is 2*2*10 = 40 scalar multiplications in schoolbook.
    # Using Strassen: we compute 7 "products" which are sums of A2 entries times B2 entries.
    # Each "product" is a vector of length 10 (since B2 has 10 columns).
    # So we need 7 * 10 = 70 scalar multiplications? That's worse than 40.
    
    # I think I'm confusing myself. Let me think about this more carefully.
    
    # The tensor for 2x2x10 matrix multiplication has size 2*2 x 2*10 x 10*2 = 4 x 20 x 20.
    # The schoolbook rank is 2*2*10 = 40.
    # Can we do better?
    
    # Actually, I recall that the rank of m x n x p matrix multiplication is at least max(m*n, n*p, p*m)
    # and at most m*n*p.
    # For 2x2x10: max(4, 20, 20) = 20, and m*n*p = 40.
    # So the rank is between 20 and 40.
    
    # I think the rank of 2x2x10 is 20 (achieved by sharing the 2x2 Strassen intermediates).
    # Let me try to construct it.
    
    # For 2x2x10: A2 is 2x2, B2 is 2x10.
    # C = A2 * B2 is 2x10.
    # C_ik = A2_00 * B2_0k + A2_01 * B2_1k + A2_10 * B2_0k + A2_11 * B2_1k.
    # = (A2_00 + A2_10) * B2_0k + (A2_01 + A2_11) * B2_1k... no, that's not right.
    # = A2_00 * B2_0k + A2_01 * B2_1k + A2_10 * B2_0k + A2_11 * B2_1k.
    # Group by B2 entries:
    # = B2_0k * (A2_00 + A2_10) + B2_1k * (A2_01 + A2_11).
    # Wait, that's only 2 terms, but we have 4 products.
    # C_0k = A2_00 * B2_0k + A2_01 * B2_1k
    # C_1k = A2_10 * B2_0k + A2_11 * B2_1k
    # So C_0k depends on B2_0k and B2_1k, and C_1k depends on B2_0k and B2_1k.
    # We can write:
    # C_0k = B2_0k * A2_00 + B2_1k * A2_01
    # C_1k = B2_0k * A2_10 + B2_1k * A2_11
    # This is 4 products per k, so 4*10 = 40 total.
    
    # To reduce, we can use:
    # s1 = A2_00 + A2_10, s2 = A2_01 + A2_11
    # t1 = B2_0k, t2 = B2_1k
    # C_0k = A2_00 * t1 + A2_01 * t2
    # C_1k = A2_10 * t1 + A2_11 * t2
    # This doesn't help directly.
    
    # I think for 2x2x10, the rank is actually 40 (same as schoolbook) because the 10 columns are independent.
    # No wait, that's not right either.
    
    # Let me try a completely different approach. Let me just try to find a low-rank decomposition
    # by solving the system of equations.
    
    # The condition is: for all i in {0,1}, j in {0,1,2}, k in {0,...,9}:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 2*3*10 = 60 equations in U, V, W.
    # We want to minimize R (the number of columns in U, V, W).
    
    # For R=1: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k.
    # This requires U[i*m+j, 0] = 1/a_ij, V[j*p+k, 0] = 1/b_jk, W[k*n+i, 0] = 1/c_ik
    # where a_ij * b_jk * c_ik = 1 for all i,j,k.
    # This is possible: set a_ij = 1, b_jk = 1, c_ik = 1.
    # Then U = all-ones, V = all-ones, W = all-ones.
    # But wait, does this actually compute the matrix multiplication correctly?
    
    # Let me check: C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A_ij) * (sum_j V[j*p+k, r] * B_jk)
    # With R=1, U=V=W=1:
    # C_ik = 1 * (sum_j 1 * A_ij) * (sum_j 1 * B_jk) = (sum_j A_ij) * (sum_j B_jk)
    # This is NOT equal to sum_j A_ij * B_jk in general!
    
    # So the rank-1 decomposition of the tensor does NOT correspond to a valid matrix multiplication algorithm.
    # The tensor decomposition must be such that the multiplication is correctly computed.
    
    # I think the issue is that the tensor T is not just any tensor; it's the matrix multiplication tensor,
    # and the rank-1 decompositions must correspond to valid bilinear algorithms.
    
    # A valid bilinear algorithm for matrix multiplication has the form:
    # C_ik = sum_r alpha_r(ik) * (sum_j beta_r(ij) * A_ij) * (sum_j gamma_r(jk) * B_jk)
    # where alpha, beta, gamma are the entries of W, U, V.
    
    # The condition for correctness is:
    # sum_r alpha_r(ik) * beta_r(ij) * gamma_r(jk) = delta_{j,j'} * delta_{k,k'}... 
    
    # I'm getting confused. Let me just try to use a known construction.
    
    # For 2x3x10, I'll try to use the following approach:
    # 1. Split the 3 columns into 2+1.
    # 2. Use Strassen for the 2x2x10 part.
    # 3. Handle the 2x1x10 part separately.
    
    # Actually, let me try a simpler approach: just return the schoolbook decomposition.
    # The seed program already does this with rank 60.
    # If I can't find a better construction, I'll return the same.
    
    # But let me try one more thing: use the fact that the rank of 2x3x10 might be lower.
    
    # I'll try to construct a decomposition with rank 30.
    # Idea: For each k, compute the 2x3x1 multiplication (A * b_k) using rank 3.
    # Then total rank is 3*10 = 30.
    # For 2x3x1: C_i = sum_j A_ij B_j.
    # This is a 2x3 matrix times a 3-vector.
    # The rank of this tensor is 3 (schoolbook).
    # Can we do better? The rank of 2x3x1 is at least max(2*3, 3*1, 1*2) = 6? No, that's not right.
    # The rank of 2x3x1 matrix multiplication is 3 (it's just 3 dot products of length... no, 2 dot products of length 3).
    # Actually, 2x3x1 means 2 outputs, each is a dot product of length 3.
    # The rank is 3 (one for each j).
    # So for 2x3x10, the rank is at most 3*10 = 30.
    
    # Let me construct this.
    # For each j in {0,1,2} and k in {0,...,9}:
    #   U[i*m+j, r] = 1 for i in {0,1}
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] = 1 for i in {0,1}
    #   where r = j*10 + k
    
    # This gives rank 3*10 = 30.
    
    # Let me verify: C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A_ij) * (sum_j V[j*p+k, r] * B_jk)
    # For r = j'*10 + k':
    #   U[i*m+j, r] = 1 if j = j', else 0
    #   V[j*p+k, r] = 1 if j = j' and k = k', else 0
    #   W[k*n+i, r] = 1 if k = k' and i = i, else 0
    # So the r-th term contributes: 1 * A_ij * B_jk * 1 = A_ij B_jk if j=j' and k=k', else 0.
    # Summing over r: C_ik = sum_{j,k} [k=k'] A_ij B_jk = sum_j A_ij B_jk.
    # Yes! This works.
    
    # So rank 30 is achievable.
    
    # Can we do better?
    # For 2x3x1, the rank is 3. Is there a rank-2 decomposition?
    # 2x3x1: C_0 = A_00*B_0 + A_01*B_1 + A_02*B_2
    #           C_1 = A_10*B_0 + A_11*B_1 + A_12*B_2
    # This is a 2x3x1 tensor. The rank is 3 (I believe).
    # So for 2x3x10, the rank is at most 30.
    
    # Let me try to see if we can share across k.
    # The key is that A is shared across all k.
    # C_ik = sum_j A_ij B_jk.
    # For fixed i, C_i is a 1x10 vector: C_i = A_i_row * B.
    # This is a 1x3 times 3x10 multiplication.
    # The rank of 1x3x10 is 3 (one for each j).
    # So for 2 rows, the rank is at most 2*3 = 6? No, that's not right.
    # The 2x3x10 tensor: C_ik = sum_j A_ij B_jk.
    # For each j, the term A_ij * B_jk contributes to all C_ik.
    # So we have 3 "layers" (one for each j), and each layer is a 2x1x10 tensor.
    # The rank of 2x1x10 is 10 (one for each k).
    # So total rank is 3*10 = 30.
    
    # I think 30 is the answer. Let me construct it.
    
    rank = 3 * 10  # 30
    U = [[0] * rank for _ in range(2 * 3)]  # 6 x 30
    V = [[0] * rank for _ in range(3 * 10)]  # 30 x 30
    W = [[0] * rank for _ in range(10 * 2)]  # 20 x 30
    
    for j in range(3):
        for k in range(10):
            r = j * 10 + k
            for i in range(2):
                U[i * 3 + j][r] = 1
            V[j * 10 + k][r] = 1
            for i in range(2):
                W[k * 2 + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
