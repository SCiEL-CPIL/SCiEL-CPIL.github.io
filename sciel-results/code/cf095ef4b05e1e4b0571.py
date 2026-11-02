import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill
    
    # Build the 3x5x5 matrix multiplication tensor as a 3D array
    # T[i,j,k] = 1 if the (i,j) element of A and (j,k) element of B
    # contribute to the (i,k) element of C
    # We need to find rank-1 decompositions: sum of U[i*m+j]*V[j*p+k]*W[k*n+i] = 1 for all (i,j,k)
    
    # The tensor T has shape (n*m, m*p, p*n) = (15, 25, 15)
    # T[a,b,c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if the indices are consistent: 
    # a = i*m+j means i = a//m, j = a%m
    # b = j*p+k means j2 = b//p, k = b%p, and we need j2 == j
    # c = k*n+i means k2 = c//n, i2 = c%n, and we need k2 == k and i2 == i
    
    # Actually, the tensor is: T[i,j,k] = 1 for all valid (i,j,k)
    # The 3D tensor is T[a,b,c] = 1 iff:
    #   a = i*m + j for some i,j
    #   b = j*p + k for some j,k (same j as above)
    #   c = k*n + i for some k,i (same k and i as above)
    # Otherwise T[a,b,c] = 0
    
    # Let's verify: for a given (i,j,k), the contribution to C[i,k] is A[i,j]*B[j,k]
    # So the tensor element at position (i*m+j, j*p+k, k*n+i) should be 1
    # All other elements should be 0
    
    # Schoolbook: rank = n*m*p = 75
    # We want to find a lower rank decomposition
    
    # Strategy: Use a block-partitioned approach
    # Partition the 3x5x5 tensor into blocks:
    # - Top 2 rows of A (i=0,1) with all 5 columns, times all 5x5 of B
    # - Bottom 1 row of A (i=2) with all 5 columns, times all 5x5 of B
    
    # For the top 2x5x5 block, we can try to share bilinear forms between the two output rows
    # For the bottom 1x5x5 block, we use schoolbook (rank 5)
    
    # Let's think about the top 2x5x5 block more carefully.
    # The tensor for rows i=0,1 has shape (2*5, 25, 15) = (10, 25, 15)
    # We need: sum_c U[a,c]*V[b,c]*W[c_elem,c] = T[a,b,c_elem] for all valid (a,b,c_elem)
    
    # Actually, let's use a different approach. Let's try to find a rank-19 decomposition
    # for the 2x5x5 block by exploiting structure.
    
    # The 2x5x5 matrix multiplication tensor can be viewed as:
    # For each k (column of B / row of C), we have a 2x5x1 sub-tensor
    # That's 5 separate 2x5x1 tensors, each of rank at most 2 (since it's just two scalars times B's row)
    # But we can share the B factors across k values.
    
    # Let's think of it as: C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1
    # This is 2*5 = 10 outputs, each being a dot product of a 5-vector A[i,:] with a 5-vector B[:,k]
    
    # A rank-1 term is: u_i * v_j * w_k (with appropriate index structure)
    # We need to find the minimum number of such terms that sum to give all 10 dot products.
    
    # One approach: Use the fact that we can write A[i,:] as a sum of rank-1 updates
    # But the standard approach is to use fast matrix multiplication.
    
    # For 2x5, we can pad to 4x4 (or use other techniques)
    # 4x4x4 has rank 21 (Strassen-like or better)
    # But we need 2x5x5, not 4x4x4
    
    # Let me try a different strategy: use a greedy/local search approach
    # Start from schoolbook and try to merge pairs of terms
    
    # Actually, let's try a more systematic approach.
    # The tensor T for 3x5x5 has 75 ones (one for each (i,j,k) triple).
    # We want to cover these 75 positions with rank-1 tensors (each rank-1 tensor is a 
    # 3D array of outer product form u_i * v_j * w_k that has 1s only at positions 
    # consistent with the matrix multiplication structure).
    
    # A rank-1 tensor for matrix multiplication has the form:
    # It's defined by a vector u over (i,j) indices, v over (j,k) indices, w over (k,i) indices
    # such that the tensor is u[i,j] * v[j,k] * w[k,i]
    # For this to be a valid decomposition of the MM tensor, we need:
    # sum_rank u_r[i,j] * v_r[j,k] * w_r[k,i] = 1 for all valid (i,j,k)
    # and = 0 for all invalid (i,j,k)
    
    # The "invalid" positions are those where the indices don't match:
    # e.g., u has support on (i,j) but v has support on (j',k) with j' != j
    # The product u[i,j]*v[j',k]*w[k',i'] is only non-zero at position (i*m+j, j'*p+k, k'*n+i')
    # which is invalid if j' != j or k' != k or i' != i
    
    # So each rank-1 term is a "consistent" 3D array where:
    # u has support on some set S_u of (i,j) pairs
    # v has support on some set S_v of (j,k) pairs  
    # w has support on some set S_w of (k,i) pairs
    # And the term is non-zero only where all three indices are consistent
    
    # For a single rank-1 term with u having support {(i1,j1), (i2,j2), ...}, 
    # v having support {(j1,k1), (j1,k2), ...}, w having support {(k1,i1), (k2,i1), ...}
    # The term contributes to positions (i1,j1,k1), (i1,j1,k2), (i2,j2,k1), etc.
    # where all indices are consistent.
    
    # This is getting complex. Let me try a computational approach:
    # 1. Start with schoolbook (rank 75)
    # 2. Try to find pairs of terms that can be merged into one
    # 3. Iterate until no more merges possible
    
    # A pair of rank-1 terms can be merged if there exist rank-1 terms that sum to the same thing
    # Specifically, two rank-1 terms u1*v1*w1 + u2*v2*w2 can sometimes be written as a single rank-1 term
    # This happens when the 3D arrays are "proportional" in some sense
    
    # Actually, two rank-1 terms can be merged into one rank-1 term only if they are already
    # proportional (i.e., one is a scalar multiple of the other), which is trivial.
    # More generally, we can try to replace k terms with fewer terms.
    
    # Let me try a different approach: use known fast matrix multiplication results.
    # For 2xN matrices, the rank is at most 2N - 1 (using Strassen-like methods)
    # But I'm not sure about the exact result.
    
    # Let me try yet another approach: use the fact that the problem is small (75 terms)
    # and do a local search / simulated annealing to find a better decomposition.
    
    # Actually, let me reconsider the block approach.
    # Partition into:
    # Block 1: i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4} -> 2x5x5 sub-tensor
    # Block 2: i in {2}, j in {0,1,2,3,4}, k in {0,1,2,3,4} -> 1x5x5 sub-tensor
    
    # Block 2 is rank 5 (schoolbook)
    # Block 1: 2x5x5. Let's see if we can do better than rank 25 (schoolbook).
    
    # For 2xN x NxN matrix multiplication, the rank is at most 2N-1 for N>=2?
    # Let me think... For 2x2x2, rank is 7 (Strassen). For 2x3x3, what's the rank?
    # 
    # Actually, I recall that for m x n x p matrix multiplication, there are known bounds.
    # For 2x5x5, let me try to construct a decomposition.
    
    # Let A be 2x5, B be 5x5, C = AB be 2x5.
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0,1,2,3,4
    
    # We can write this as:
    # C[0,:] = A[0,:] @ B (dot product of row 0 of A with each column of B)
    # C[1,:] = A[1,:] @ B (dot product of row 1 of A with each column of B)
    
    # One way to decompose: for each j, we have A[0,j]*B[j,:] and A[1,j]*B[j,:]
    # That's 5 rank-1 terms for row 0 and 5 for row 1, total 10. But we can share B[j,:].
    
    # Actually, each "term" in the tensor decomposition is a 3D rank-1 array.
    # For the 2x5x5 sub-tensor, a rank-1 term is defined by:
    # u over (i,j) for i in {0,1}, j in {0,1,2,3,4}
    # v over (j,k) for j in {0,1,2,3,4}, k in {0,1,2,3,4}
    # w over (k,i) for k in {0,1,2,3,4}, i in {0,1}
    
    # The term contributes u[i,j]*v[j,k]*w[k,i] to position (i,j,k)
    
    # If u has support on a single (i,j) pair, v on a single (j',k) pair, w on a single (k',i') pair,
    # then the term is non-zero only if i=i', j=j', k=k', and the value is u[i,j]*v[j,k]*w[k,i].
    # This is a schoolbook term with rank 1.
    
    # To get a better decomposition, we need u, v, w to have larger support.
    # For example, if u[0,0]=1, u[1,0]=1, v[0,0]=1, v[0,1]=1, w[0,0]=1, w[1,0]=1,
    # then the term contributes to:
    # (0,0,0): 1*1*1 = 1
    # (0,0,1): 1*1*1 = 1
    # (1,0,0): 1*1*1 = 1
    # (1,0,1): 1*1*1 = 1
    # That's 4 positions covered by one rank-1 term!
    
    # So the idea is to find large "rectangular" supports for u, v, w that cover many positions.
    
    # Let me try to construct a decomposition for 2x5x5.
    # 
    # Idea: Partition the 5 columns of A into groups, and use Strassen-like ideas.
    # 
    # Actually, let me try a simpler approach: use the fact that we can write
    # A[0,:] and A[1,:] as linear combinations of a few basis vectors, and similarly for B's columns.
    
    # Let me try: define 3 vectors a0, a1, a2 in R^5 such that A[0,:] = a0 + a1 and A[1,:] = a0 + a2
    # (This requires that A[0,:] - A[1,:] = a1 - a2, which is always possible with a0=0, a1=A[0,:], a2=-A[1,:])
    # Wait, that doesn't help directly.
    
    # Let me try another approach. Define:
    # x = A[0,:] + A[1,:]
    # y = A[0,:] - A[1,:]
    # Then A[0,:] = (x+y)/2, A[1,:] = (x-y)/2
    
    # C[0,:] = (x+y)/2 @ B = (x@B + y@B)/2
    # C[1,:] = (x-y)/2 @ B = (x@B - y@B)/2
    
    # So we need to compute x@B and y@B, which are two 1x5 vectors, each being a dot product of a 5-vector with 5 columns of B.
    # Each dot product can be done with 5 rank-1 terms (schoolbook), so x@B needs 5 terms and y@B needs 5 terms.
    # But wait, x and y are fixed linear combinations of A's rows, so in the tensor decomposition,
    # u for x@B would have u[0,j] = 1 and u[1,j] = 1 for all j (since x = A[0,:]+A[1,:])
    # and u for y@B would have u[0,j] = 1 and u[1,j] = -1 for all j
    
    # So:
    # For x@B: u_x[0,j] = 1, u_x[1,j] = 1 for all j. Then for each k, v[j,k] = 1 for all j, and w[k,0] = 1/2, w[k,1] = 1/2
    # Wait, that's not right. Let me think more carefully.
    
    # The term for x@B at position (i,j,k) should be:
    # u_x[i,j] * v[j,k] * w[k,i]
    # We want this to contribute to C[0,k] and C[1,k] equally.
    # C[0,k] gets x[j]*B[j,k] summed over j, with coefficient 1/2
    # C[1,k] gets x[j]*B[j,k] summed over j, with coefficient 1/2
    
    # So for the x part:
    # u_x[i,j] = 1 for all i,j (both rows, all columns)
    # v[j,k] = 1 for all j,k
    # w[k,0] = 1/2, w[k,1] = 1/2 for all k
    
    # But this is a single rank-1 term? No, because w depends on k.
    # Actually, w is a 5x2 matrix (k x i), and for it to be rank-1, w[k,i] = alpha[k]*beta[i].
    # If w[k,0] = 1/2 and w[k,1] = 1/2 for all k, then alpha[k] = 1/2 and beta[i] = 1, which works.
    # So w = (1/2) * [1,1]^T * [1,1] = all entries 1/2.
    # And u_x = all ones (2x5), which is rank 1: [1,1]^T * [1,1,1,1,1]
    # And v = all ones (5x5), which is rank 1: [1,1,1,1,1]^T * [1,1,1,1,1]
    
    # So the x part is ONE rank-1 term! Similarly, the y part is one rank-1 term.
    # Total for 2x5x5: 2 rank-1 terms?
    
    # Wait, that can't be right. Let me check.
    # The x term: u_x[i,j]*v[j,k]*w[k,i] = 1*1*(1/2) = 1/2 for all (i,j,k)
    # This contributes 1/2 to every position in the 2x5x5 tensor.
    # The y term: u_y[i,j]*v[j,k]*w_y[k,i] where u_y[0,j]=1, u_y[1,j]=-1, v all ones, w_y[k,0]=1/2, w_y[k,1]=-1/2
    # w_y[k,i] = (1/2)*[1,-1][i] = alpha[k]*beta[i] with alpha[k]=1/2, beta=[1,-1]
    # So w_y = (1/2)*[1,1,1,1,1]^T * [1,-1]
    # u_y = [1,1,1,1,1]^T... no, u_y is 2x5: u_y[0,j]=1, u_y[1,j]=-1 = [1,-1]^T * [1,1,1,1,1]
    # v = all ones = [1,1,1,1,1]^T * [1,1,1,1,1]
    # So the y term at (i,j,k) = [1,-1][i] * 1 * (1/2)*[1,-1][i] = (1/2)*[1,-1][i]^2 = 1/2 for i=0, -1/2 for i=1
    # Wait: u_y[i,j] = [1,-1][i], v[j,k] = 1, w_y[k,i] = (1/2)*[1,-1][i]
    # Product = [1,-1][i] * 1 * (1/2)*[1,-1][i] = (1/2)*[1,-1][i]^2 = 1/2 (since [1,-1][i]^2 = 1)
    # So the y term also contributes 1/2 to every position? That's not right.
    
    # I think I'm confusing myself. Let me be more careful.
    
    # The tensor T[i,j,k] for 2x5x5 is 1 for all i in {0,1}, j in {0,1,2,3,4}, k in {0,1,2,3,4}.
    # We want sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] = 1 for all (i,j,k).
    
    # Term 1 (x part): u_1[i,j] = 1 for all i,j. v_1[j,k] = 1 for all j,k. w_1[k,i] = 1/2 for all k,i.
    # Contribution at (i,j,k): 1*1*(1/2) = 1/2.
    
    # Term 2 (y part): u_2[i,j] = [1,-1][i] for all j. v_2[j,k] = 1 for all j,k. w_2[k,i] = (1/2)*[1,-1][i].
    # Contribution at (i,j,k): [1,-1][i]*1*(1/2)*[1,-1][i] = (1/2)*([1,-1][i])^2 = 1/2.
    
    # Total: 1/2 + 1/2 = 1. Correct!
    
    # But wait, this decomposition has rank 2 for 2x5x5? That seems too good to be true.
    # Let me double-check by considering what the actual matrix multiplication looks like.
    
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # For i=0: C[0,k] = sum_j A[0,j]*B[j,k]
    # For i=1: C[1,k] = sum_j A[1,j]*B[j,k]
    
    # In the tensor decomposition, the term u[i,j]*v[j,k]*w[k,i] should represent
    # the contribution to C[i,k] from A[i,j]*B[j,k].
    # 
    # Actually, I think the issue is that in the tensor decomposition, each rank-1 term
    # corresponds to a "bilinear form" that is applied to A and B to produce a contribution to C.
    # The term u[i,j]*v[j,k]*w[k,i] means:
    # - u[i,j] is the coefficient of A[i,j]
    # - v[j,k] is the coefficient of B[j,k]
    # - w[k,i] is the coefficient of the product A[i,j]*B[j,k] in C[i,k]
    #
    # But this is only valid if the product A[i,j]*B[j,k] actually contributes to C[i,k].
    # In standard matrix multiplication, A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k).
    #
    # Now, in the rank-1 decomposition, we have:
    # T[i,j,k] = sum_r u_r[i,j]*v_r[j,k]*w_r[k,i]
    #
    # This means that for each r, the "slice" of the tensor at rank r is u_r (2x5), v_r (5x5), w_r (5x2).
    # And the tensor is the sum of outer products u_r ⊗ v_r ⊗ w_r (with appropriate index structure).
    #
    # Now, for the decomposition to be valid, we need:
    # sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] = 1 for all i,j,k
    #
    # My proposed terms:
    # Term 1: u_1 = all ones (2x5), v_1 = all ones (5x5), w_1 = all 1/2 (5x2)
    # Term 2: u_2[i,j] = [1,-1][i], v_2 = all ones (5x5), w_2[k,i] = (1/2)*[1,-1][i]
    #
    # Check: sum_r u_r[i,j]*v_r[j,k]*w_r[k,i]
    # = 1*1*(1/2) + [1,-1][i]*1*(1/2)*[1,-1][i]
    # = 1/2 + (1/2)*([1,-1][i])^2
    # = 1/2 + 1/2 = 1 ✓
    #
    # So this seems to work! But is this a valid matrix multiplication decomposition?
    # The issue is that in actual matrix multiplication, the bilinear forms must be such that
    # the intermediate products can be computed and combined to give C.
    #
    # Let's see what the intermediate products would be:
    # For term 1: P_1 = (sum_{i,j} u_1[i,j]*A[i,j]) * (sum_{j,k} v_1[j,k]*B[j,k])
    # = (sum_{i,j} A[i,j]) * (sum_{j,k} B[j,k])
    # This is a scalar (product of two scalars).
    # And it contributes w_1[k,i]*P_1 to C[i,k] = (1/2)*(sum A)*(sum B)
    #
    # For term 2: P_2 = (sum_{i,j} u_2[i,j]*A[i,j]) * (sum_{j,k} v_2[j,k]*B[j,k])
    # = (A[0,0]+A[0,1]+A[0,2]+A[0,3]+A[0,4] - A[1,0]-A[1,1]-A[1,2]-A[1,3]-A[1,4]) * (sum B)
    # = (sum_j (A[0,j]-A[1,j])) * (sum_{j,k} B[j,k])
    # And it contributes w_2[k,i]*P_2 to C[i,k] = (1/2)*[1,-1][i]*(sum_j (A[0,j]-A[1,j]))*(sum_{j,k} B[j,k])
    #
    # Total contribution to C[i,k]:
    # (1/2)*(sum A)*(sum B) + (1/2)*[1,-1][i]*(sum_j (A[0,j]-A[1,j]))*(sum_{j,k} B[j,k])
    #
    # This is NOT equal to sum_j A[i,j]*B[j,k] in general!
    #
    # For example, if A = [[1,0,0,0,0],[0,1,0,0,0]] and B = identity, then:
    # C[0,0] = 1, C[1,0] = 0
    # My formula gives:
    # C[0,0] = (1/2)*(2)*(5) + (1/2)*(1)*(1)*(5) = 5 + 2.5 = 7.5 ≠ 1
    #
    # So my decomposition is WRONG. The issue is that the tensor decomposition
    # T[i,j,k] = sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] only ensures that the tensor is decomposed,
    # but it doesn't ensure that the bilinear forms compute the correct matrix product.
    #
    # Actually, I think I was confused. The tensor T[i,j,k] = 1 means that the (i,j) element of A
    # and the (j,k) element of B contribute to the (i,k) element of C. The rank-1 decomposition
    # T[i,j,k] = sum_r u_r[i,j]*v_r[j,k]*w_r[k,i] is a valid tensor decomposition IF AND ONLY IF
    # for all A and B, sum_r (sum_{i,j} u_r[i,j]*A[i,j]) * (sum_{j,k} v_r[j,k]*B[j,k]) * w_r[k,i] = C[i,k]
    # for all i,k.
    #
    # Wait, that's not right either. Let me think about this more carefully.
    #
    # The matrix multiplication tensor is T where T[i,j,k] = 1 if (i,j,k) is a valid index triple,
    # meaning that A[i,j]*B[j,k] contributes to C[i,k].
    #
    # A rank-1 decomposition is T[i,j,k] = sum_r u_r[i,j]*v_r[j,k]*w_r[k,i]
    # This is equivalent to saying that for all A and B:
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_r (sum_{i',j'} u_r[i',j']*A[i',j']) * (sum_{j',k'} v_r[j',k']*B[j',k']) * w_r[k,i]
    #
    # No, that's not right. The correct interpretation is:
    # C[i,k] = sum_{r} w_r[k,i] * (sum_{j} u_r[i,j]*A[i,j]) * (sum_{j} v_r[j,k]*B[j,k])
    #
    # Wait, I think the correct form is:
    # C[i,k] = sum_r (sum_j u_r[i,j]*A[i,j]) * (sum_j v_r[j,k]*B[j,k]) * w_r[k,i]
    #
    # No, that's not right. Let me look at this from first principles.
    #
    # In a rank-r tensor decomposition, we have:
    # T[i,j,k] = sum_{r=1}^r U_r[i,j] * V_r[j,k] * W_r[k,i]
    #
    # This means that the bilinear form for computing C[i,k] is:
    # C[i,k] = sum_{r=1}^r W_r[k,i] * (sum_{j} U_r[i,j] * A[i,j]) * (sum_{j} V_r[j,k] * B[j,k])
    #
    # Wait, no. The standard form is:
    # C[i,k] = sum_{r=1}^r (sum_{i',j'} U_r[i',j'] * A[i',j']) * (sum_{j',k'} V_r[j',k'] * B[j',k']) * W_r[k,i]
    #
    # No, that's a scalar times a scalar times a scalar, which is a scalar, but C[i,k] is an element of a matrix.
    #
    # I think the correct interpretation is:
    # Define intermediate values:
    # M_r = (sum_{i,j} U_r[i,j] * A[i,j]) * (sum_{j,k} V_r[j,k] * B[j,k])
    # Then C[i,k] = sum_r W_r[k,i] * M_r
    #
    # But M_r is a scalar (product of two scalars), and W_r[k,i] is a matrix element, so
    # C[i,k] = sum_r W_r[k,i] * M_r is a linear combination of scalars, which is a scalar.
    # But C is a matrix, so this doesn't make sense unless M_r depends on i and k.
    #
    # I think I've been confused about the structure. Let me reconsider.
    #
    # Actually, I think the correct form is:
    # C[i,k] = sum_r W_r[k,i] * (sum_j U_r[i,j] * A[i,j]) * (sum_j V_r[j,k] * B[j,k])
    #
    # Here, for each r, we compute:
    # a_r = sum_j U_r[i,j] * A[i,j]  (this depends on i, so it's a vector over i)
    # b_r = sum_j V_r[j,k] * B[j,k]  (this depends on k, so it's a vector over k)
    # Then C[i,k] = sum_r W_r[k,i] * a_r * b_r
    #
    # But a_r depends on i and b_r depends on k, so a_r*b_r depends on both i and k,
    # and W_r[k,i] also depends on i and k, so C[i,k] = sum_r W_r[k,i]*a_r*b_r is valid.
    #
    # Now, for this to equal sum_j A[i,j]*B[j,k], we need:
    # sum_r W_r[k,i]*a_r*b_r = sum_j A[i,j]*B[j,k] for all A, B, i, k
    #
    # This is a bilinear identity in A and B. Expanding:
    # sum_r W_r[k,i]*(sum_j U_r[i,j]*A[i,j])*(sum_j V_r[j,k]*B[j,k]) = sum_j A[i,j]*B[j,k]
    #
    # The LHS is a bilinear form in A and B. For it to equal the RHS for all A and B,
    # we need:
    # sum_r W_r[k,i]*U_r[i,j]*V_r[j,k] = delta_{valid}(i,j,k) = 1 if (i,j,k) is valid, 0 otherwise
    #
    # Where "valid" means that A[i,j]*B[j,k] contributes to C[i,k], which is always true
    # in standard matrix multiplication. So:
    # sum_r W_r[k,i]*U_r[i,j]*V_r[j,k] = 1 for all i,j,k
    #
    # This is the same as T[i,j,k] = sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] = 1 for all i,j,k.
    #
    # So my original understanding was correct! The tensor T[i,j,k] = 1 for all valid (i,j,k),
    # and we need T[i,j,k] = sum_r U_r[i,j]*V_r[j,k]*W_r[k,i].
    #
    # Now, going back to my proposed decomposition for 2x5x5:
    # Term 1: U_1 = all ones (2x5), V_1 = all ones (5x5), W_1 = all 1/2 (5x2)
    # Term 2: U_2[i,j] = [1,-1][i], V_2 = all ones (5x5), W_2[k,i] = (1/2)*[1,-1][i]
    #
    # Check: sum_r U_r[i,j]*V_r[j,k]*W_r[k,i]
    # = 1*1*(1/2) + [1,-1][i]*1*(1/2)*[1,-1][i]
    # = 1/2 + (1/2)*([1,-1][i])^2
    # = 1/2 + 1/2 = 1 ✓
    #
    # So the tensor equation is satisfied. But does this give a valid matrix multiplication algorithm?
    #
    # Let's compute C[i,k] using this decomposition:
    # C[i,k] = sum_r W_r[k,i] * (sum_j U_r[i,j]*A[i,j]) * (sum_j V_r[j,k]*B[j,k])
    #
    # Term 1: W_1[k,i] = 1/2, a_1 = sum_j A[i,j] (since U_1[i,j]=1), b_1 = sum_j B[j,k] (since V_1[j,k]=1)
    # Contribution: (1/2) * (sum_j A[i,j]) * (sum_j B[j,k])
    #
    # Term 2: W_2[k,i] = (1/2)*[1,-1][i], a_2 = sum_j [1,-1][i]*A[i,j] = [1,-1][i]*sum_j A[i,j]
    # b_2 = sum_j B[j,k] (since V_2[j,k]=1)
    # Contribution: (1/2)*[1,-1][i] * [1,-1][i]*sum_j A[i,j] * sum_j B[j,k]
    # = (1/2)*([1,-1][i])^2 * sum_j A[i,j] * sum_j B[j,k]
    # = (1/2) * sum_j A[i,j] * sum_j B[j,k]
    #
    # Total: C[i,k] = (1/2)*sum_j A[i,j]*sum_j B[j,k] + (1/2)*sum_j A[i,j]*sum_j B[j,k]
    # = sum_j A[i,j]*sum_j B[j,k]
    #
    # But this is NOT equal to sum_j A[i,j]*B[j,k] in general!
    # sum_j A[i,j]*sum_j B[j,k] = (sum_j A[i,j])*(sum_j B[j,k]) ≠ sum_j A[i,j]*B[j,k]
    #
    # So the tensor equation is satisfied, but the matrix multiplication is incorrect!
    #
    # This means that satisfying T[i,j,k] = sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] is NECESSARY but NOT SUFFICIENT
    # for a valid matrix multiplication decomposition. There must be an additional constraint.
    #
    # I think the additional constraint is that the intermediate products must be such that
    # the algorithm actually computes the correct result. In other words, the decomposition
    # must correspond to a valid sequence of additions and multiplications.
    #
    # Actually, I think I was wrong about the form of the decomposition. Let me reconsider.
    #
    # In a standard tensor decomposition of matrix multiplication, the form is:
    # C[i,k] = sum_r M_r * W_r[k,i]
    # where M_r = (sum_{i,j} U_r[i,j]*A[i,j]) * (sum_{j,k} V_r[j,k]*B[j,k])
    #
    # But M_r is a scalar, so C[i,k] = sum_r M_r*W_r[k,i] means that C is a linear combination
    # of rank-1 matrices W_r (scaled by scalars M_r).
    #
    # For this to equal AB, we need:
    # AB = sum_r M_r * W_r
    # where M_r = (U_r : A) * (V_r : B) (Frobenius inner products)
    #
    # This is only possible if AB can be written as a sum of rank-1 matrices, which is always true
    # (any matrix is a sum of rank-1 matrices), but the M_r must be bilinear in A and B.
    #
    # I think the key insight is that the tensor decomposition T[i,j,k] = sum_r U_r[i,j]*V_r[j,k]*W_r[k,i]
    # is indeed the correct condition, and my proposed decomposition should be valid.
    #
    # Let me re-examine the computation:
    # C[i,k] = sum_r W_r[k,i] * (sum_{i',j'} U_r[i',j']*A[i',j']) * (sum_{j',k'} V_r[j',k']*B[j',k'])
    #
    # Wait, I think the issue is in how I'm computing the intermediate products.
    # The intermediate product for term r is:
    # M_r = (sum_{i',j'} U_r[i',j']*A[i',j']) * (sum_{j',k'} V_r[j',k']*B[j',k'])
    # This is a scalar.
    # Then C[i,k] = sum_r M_r * W_r[k,i]
    #
    # For my decomposition:
    # M_1 = (sum_{i,j} 1*A[i,j]) * (sum_{j,k} 1*B[j,k]) = (sum A)*(sum B)
    # M_2 = (sum_{i,j} [1,-1][i]*A[i,j]) * (sum_{j,k} 1*B[j,k]) = (sum_j(A[0,j]-A[1,j]))*(sum B)
    #
    # C[i,k] = W_1[k,i]*M_1 + W_2[k,i]*M_2
    # = (1/2)*(sum A)*(sum B) + (1/2)*[1,-1][i]*(sum_j(A[0,j]-A[1,j]))*(sum B)
    #
    # For i=0: C[0,k] = (1/2)*(sum A)*(sum B) + (1/2)*(sum_j(A[0,j]-A[1,j]))*(sum B)
    # = (1/2)*(sum A + sum_j(A[0,j]-A[1,j]))*(sum B)
    # = (1/2)*(2*sum_j A[0,j])*(sum B)
    # = (sum_j A[0,j])*(sum B)
    #
    # For i=1: C[1,k] = (1/2)*(sum A)*(sum B) + (1/2)*(-1)*(sum_j(A[0,j]-A[1,j]))*(sum B)
    # = (1/2)*(sum A - sum_j(A[0,j]-A[1,j]))*(sum B)
    # = (1/2)*(2*sum_j A[1,j])*(sum B)
    # = (sum_j A[1,j])*(sum B)
    #
    # So C[i,k] = (sum_j A[i,j])*(sum B) for all i,k.
    # This is NOT equal to sum_j A[i,j]*B[j,k] in general.
    #
    # So my decomposition is indeed invalid for matrix multiplication, even though it satisfies the tensor equation.
    #
    # I must be misunderstanding the tensor decomposition condition. Let me look at this differently.
    #
    # I think the issue is that the tensor T[i,j,k] for matrix multiplication is not simply 1 for all (i,j,k).
    # The tensor encodes the entire bilinear map, not just the indices.
    #
    # Actually, I think the correct tensor is:
    # T[i,j,k] = 1 if the (i,j) element of A and the (j,k) element of B contribute to the (i,k) element of C.
    # In standard matrix multiplication, this is true for all valid (i,j,k), so T[i,j,k] = 1.
    #
    # But the rank-1 decomposition T[i,j,k] = sum_r U_r[i,j]*V_r[j,k]*W_r[k,i] should be valid
    # if and only if the matrix multiplication can be computed with rank r.
    #
    # I'm clearly making an error somewhere. Let me just use the schoolbook decomposition and try to improve it
    # using a local search approach, rather than trying to construct a clever decomposition from scratch.
    
    # Schoolbook decomposition for 3x5x5: rank 75
    # Each term r corresponds to a specific (i,j,k) triple:
    # U_r[i*m+j] = 1 if (i,j) matches, 0 otherwise
    # V_r[j*p+k] = 1 if (j,k) matches, 0 otherwise  
    # W_r[k*n+i] = 1 if (k,i) matches, 0 otherwise
    
    # Let me represent the decomposition as a list of terms, where each term is (i, j, k)
    # and the coefficient is 1.
    
    # Start with schoolbook
    terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # Now try to merge pairs of terms
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exists a rank-1 term
    # that covers both positions.
    #
    # A rank-1 term is defined by U (n x m), V (m x p), W (p x n)
    # It covers position (i,j,k) if U[i,j]*V[j,k]*W[k,i] != 0
    #
    # For two schoolbook terms to be merged into one rank-1 term, we need:
    # U[i1,j1]*V[j1,k1]*W[k1,i1] = 1
    # U[i2,j2]*V[j2,k2]*W[k2,i2] = 1
    # U[i,j]*V[j,k]*W[k,i] = 0 for all other (i,j,k)
    #
    # This is a system of equations. Let me try to find such U, V, W.
    
    # For simplicity, let me restrict to terms where U, V, W have at most 2 non-zero entries each.
    
    def can_merge(term1, term2, all_terms):
        """Check if two terms can be merged into a single rank-1 term."""
        i1, j1, k1 = term1
        i2, j2, k2 = term2
        
        # The rank-1 term must cover (i1,j1,k1) and (i2,j2,k2)
        # and no other terms in all_terms
        
        # Try to find U, V, W with small support
        # U must have non-zero entries at (i1,j1) and (i2,j2) [or a subset]
        # V must have non-zero entries at (j1,k1) and (j2,k2) [or a subset]
        # W must have non-zero entries at (k1,i1) and (k2,i2) [or a subset]
        
        # Case 1: U has exactly {(i1,j1), (i2,j2)}, V has exactly {(j1,k1), (j2,k2)}, W has exactly {(k1,i1), (k2,i2)}
        # Then U[i1,j1]*V[j1,k1]*W[k1,i1] = 1 and U[i2,j2]*V[j2,k2]*W[k2,i2] = 1
        # And U[i,j]*V[j,k]*W[k,i] = 0 for all other (i,j,k)
        
        # Let U[i1,j1] = a, U[i2,j2] = b, V[j1,k1] = c, V[j2,k2] = d, W[k1,i1] = e, W[k2,i2] = f
        # Then ac*e = 1 and bd*f = 1
        # Also, for any other (i,j,k) in all_terms, we need U[i,j]*V[j,k]*W[k,i] = 0
        
        # Check if there are any other terms that would be covered
        covered = {(i1,j1,k1), (i2,j2,k2)}
        for (i,j,k) in all_terms:
            if (i,j,k) not in covered:
                # Check if this term would be covered by the proposed rank-1 term
                # U[i,j] is non-zero only if (i,j) in {(i1,j1), (i2,j2)}
                # V[j,k] is non-zero only if (j,k) in {(j1,k1), (j2,k2)}
                # W[k,i] is non-zero only if (k,i) in {(k1,i1), (k2,i2)}
                if (i,j) in {(i1,j1), (i2,j2)} and (j,k) in {(j1,k1), (j2,k2)} and (k,i) in {(k1,i1), (k2,i2)}:
                    return False
        
        return True
    
    # Try all pairs of terms and merge if possible
    merged = True
    while merged:
        merged = False
        new_terms = list(terms)
        for idx1 in range(len(new_terms)):
            for idx2 in range(idx1+1, len(new_terms)):
                if can_merge(new_terms[idx1], new_terms[idx2], new_terms):
                    # Merge the two terms
                    # Replace with a single term that represents the merged rank-1 term
                    # For now, just remove both and add a placeholder
                    del new_terms[idx2]
                    del new_terms[idx1]
                    # The merged term is represented by the pair, but we need to track it
                    # For simplicity, let's just count the reduction
                    merged = True
                    break
            if merged:
                break
        terms = new_terms
    
    rank = len(terms)
    
    # Build the U, V, W matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (i, j, k) in enumerate(terms):
        U[i*m+j][r] = 1.0
        V[j*p+k][r] = 1.0
        W[k*n+i][r] = 1.0
    
    # Verify
    if time.time() - start_time < DEADLINE - 5:
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                    expected = 1.0
                    err = abs(val - expected)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-10:
            # Fallback to schoolbook
            total_rank_fb = n * m * p
            U_fb = [[0.0] * total_rank_fb for _ in range(n * m)]
            V_fb = [[0.0] * total_rank_fb for _ in range(m * p)]
            W_fb = [[0.0] * total_rank_fb for _ in range(p * n)]
            col_fb = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][col_fb] = 1.0
                        V_fb[j*p+k][col_fb] = 1.0
                        W_fb[k*n+i][col_fb] = 1.0
                        col_fb += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
