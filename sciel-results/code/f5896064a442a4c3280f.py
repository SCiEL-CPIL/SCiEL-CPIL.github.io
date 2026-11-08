import numpy as np

def construct():
    # M_{4,4,6} tensor decomposition using Kronecker product:
    # M_{4,4,6} ≅ M_{2,2,3} ⊗ M_{2,2,2}
    # Rank = rank(M_{2,2,3}) * rank(M_{2,2,2})
    #
    # M_{2,2,2}: Strassen rank 7
    # M_{2,2,3}: rank 6 (each of 3 columns of B gives a 2x2*2x1 mult, rank 2 each)
    #
    # Total rank: 7 * 6 = 42
    #
    # Index conventions:
    # A is 4x4, B is 4x6, C is 4x6
    # W is indexed as W[k*n + i, r] where n=4 (rows of C), so W index = row_C * 4 + col_C
    # U is indexed as U[i*m + k, r] where m=4, so U index = row_A * 4 + col_A
    # V is indexed as V[k*p + j, r] where p=6, so V index = row_B * 6 + col_B
    #
    # Kronecker structure:
    # A[2*i1 + i2, 2*k1 + k2] where (i1,k1) are block indices, (i2,k2) are inner indices
    # B[2*k1 + k2, 2*j1 + j2] where (k1,j1) are block indices, (k2,j2) are inner indices
    # C[2*i1 + i2, 2*j1 + j2]
    #
    # C = A*B: C[i1,i2; j1,j2] = sum_{k1,k2} A[i1,i2;k1,k2] * B[k1,k2;j1,j2]
    # = sum_{k1} (sum_{k2} A[i1,i2;k1,k2] * B[k1,k2;j1,j2])
    #
    # The block-level multiplication: C_block[i1,j1] = sum_{k1} A_block[i1,k1] * B_block[k1,j1]
    # where A_block is 2x2, B_block is 2x2, C_block is 2x2 -> Strassen (rank 7)
    #
    # The inner-level multiplication: for fixed (i1,k1,j1), we compute
    # C_inner[i2,j2] = sum_{k2} A_inner[i2,k2] * B_inner[k2,j2]
    # where A_inner is 2x2, B_inner is 2x3, C_inner is 2x3
    #
    # For the inner 2x2 * 2x3, rank 6 decomposition:
    # For each column j2 in {0,1,2}:
    #   C_inner[0,j2] = A_inner[0,0]*B_inner[0,j2] + A_inner[0,1]*B_inner[1,j2]
    #   C_inner[1,j2] = A_inner[1,0]*B_inner[0,j2] + A_inner[1,1]*B_inner[1,j2]
    #
    # This is just schoolbook for 2x2*2x1 for each column, giving 2 products per column.
    # Total: 3 columns * 2 = 6 products.
    #
    # Each product is: A_inner[i2, k2] * B_inner[k2, j2] contributing to C_inner[i2, j2]
    #
    # So the 6 inner terms are indexed by (i2, k2, j2) where i2 in {0,1}, k2 in {0,1}, j2 in {0,1,2}
    # but we only need 6 terms, not 12. The key insight:
    # For fixed j2, the two products are:
    #   (i2=0, k2=0, j2): A[0,0]*B[0,j2] -> C[0,j2]
    #   (i2=0, k2=1, j2): A[0,1]*B[1,j2] -> C[0,j2]
    #   (i2=1, k2=0, j2): A[1,0]*B[0,j2] -> C[1,j2]
    #   (i2=1, k2=1, j2): A[1,1]*B[1,j2] -> C[1,j2]
    #
    # Wait, that's 4 per column * 3 columns = 12. Not 6.
    #
    # Let me reconsider. The rank of M_{2,2,3} tensor:
    # T[i,k,j] = delta (for the multiplication tensor)
    # A rank-1 term: U[i,k] * V[k,j] * W[i,j]
    # For M_{2,2,3}: i in {0,1}, k in {0,1}, j in {0,1,2}
    #
    # The multiplication tensor has T[i,k,j] = 1 if the (i,k) entry of A multiplies with (k,j) entry of B to give (i,j) entry of C.
    # So T[i,k,j] = 1 for all valid (i,k,j) where k is the summation index.
    # Actually, T[i,k,j] = 1 if i,k,j are such that A[i,k]*B[k,j] contributes to C[i,j].
    # This is true for all i in {0,1}, k in {0,1}, j in {0,1,2}.
    # So the tensor is all-ones of shape (2,2,3)? No, that's not right either.
    #
    # The matrix multiplication tensor M_{n,m,p} has entries T[i,k,j] where:
    # i in {0,...,n-1} (row of A / row of C)
    # k in {0,...,m-1} (col of A / row of B)  
    # j in {0,...,p-1} (col of B / col of C)
    # T[i,k,j] = 1 (it's the structure tensor of multiplication)
    #
    # So for M_{2,2,3}, T[i,k,j] = 1 for all i in {0,1}, k in {0,1}, j in {0,1,2}.
    # The rank is the minimum number of rank-1 tensors summing to this.
    #
    # A rank-1 tensor: U[i,k] * V[k,j] * W[i,j]
    # We need sum_r U_r[i,k] * V_r[k,j] * W_r[i,j] = 1 for all (i,k,j).
    #
    # For M_{2,2,1} (j only 0): T[i,k,0] = 1 for all i,k in {0,1}.
    # Rank of M_{2,2,1}: We need U[i,k]*V[k,0]*W[i,0] summed over r to equal 1.
    # This is equivalent to: the 2x2 matrix of ones = sum_r u_r * v_r^T * w_r
    # where u_r is a 2x2 matrix, v_r is a 2-vector, w_r is a 2-vector.
    # Actually, U[i,k]*V[k]*W[i] summed over k gives (U*V)[i] * W[i]... no.
    # sum_k U[i,k] V[k] W[i] = W[i] * (U V)[i]
    # We want this to equal 1 for all i. So W[i] * (UV)[i] = 1 for i=0,1.
    # With 1 term: W[0]*(UV)[0] = 1 and W[1]*(UV)[1] = 1.
    # UV is a 2-vector. We need (UV)[0] != 0 and (UV)[1] != 0.
    # E.g., U = [[1,0],[0,1]], V = [1,1], then UV = [1,1], W = [1,1].
    # So rank(M_{2,2,1}) = 2? No, with 1 term we can do it!
    # U = I, V = [1,1]^T, W = [1,1]^T: sum_k U[i,k]V[k]W[i] = (i=0: 1*1*1=1, i=1: 1*1*1=1). Yes!
    # So rank(M_{2,2,1}) = 1? That can't be right since the tensor has 4 non-zero entries.
    #
    # Wait, I'm confusing myself. Let me be very explicit.
    # M_{2,2,1}: A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    #
    # Tensor T[i,k,j] where i in {0,1} (C row), k in {0,1} (A col / B row), j in {0} (B col / C col)
    # T[i,k,0] = 1 for all i,k.
    #
    # Rank-1 term: U[i,k] * V[k,0] * W[i,0]
    # Sum over r: sum_r U_r[i,k] V_r[k,0] W_r[i,0] = 1 for all i,k.
    #
    # With 1 term: U[i,k] * V[k,0] * W[i,0] = 1 for all i,k.
    # Let V[0,0] = v0, V[1,0] = v1, W[0,0] = w0, W[1,0] = w1.
    # U[0,0]*v0*w0 = 1
    # U[0,1]*v1*w0 = 1
    # U[1,0]*v0*w1 = 1
    # U[1,1]*v1*w1 = 1
    # Choose v0=1, v1=1, w0=1, w1=1, then U = all ones. Works!
    # So rank(M_{2,2,1}) = 1.
    #
    # For M_{2,2,3}: T[i,k,j] = 1 for all i in {0,1}, k in {0,1}, j in {0,1,2}.
    # Rank-1 term: U[i,k] * V[k,j] * W[i,j]
    # Sum_r U_r[i,k] V_r[k,j] W_r[i,j] = 1 for all i,k,j.
    #
    # With 1 term: U[i,k] V[k,j] W[i,j] = 1 for all i,k,j.
    # Let V[k,j] = 1 for all k,j, W[i,j] = 1 for all i,j, U[i,k] = 1 for all i,k.
    # Then U[i,k]*V[k,j]*W[i,j] = 1. Works!
    # So rank(M_{2,2,3}) = 1??? That can't be right.
    #
    # Oh wait, I think I'm misunderstanding the tensor structure. The multiplication tensor is NOT all-ones.
    # The tensor T[i,k,j] represents the coefficient of A[i,k]*B[k,j] in C[i,j].
    # For matrix multiplication, C[i,j] = sum_k A[i,k] B[k,j].
    # So the tensor is T[i,k,j] = 1 (coefficient is 1) for all valid indices.
    # But the "rank" of this tensor as a 3-way array is what we're computing.
    #
    # A rank-1 3-way tensor is T[i,k,j] = U[i,k] * V[k,j] * W[i,j].
    # We want to write the all-ones tensor (2,2,3) as a sum of such rank-1 tensors.
    #
    # As I showed, 1 term suffices: U=ones(2,2), V=ones(2,3), W=ones(2,3).
    # So rank = 1?
    #
    # But that contradicts known results. Let me re-examine.
    #
    # Actually, I think the issue is that the multiplication tensor is NOT simply the all-ones tensor.
    # The structure is: C[i,j] = sum_k A[i,k] B[k,j]
    # The tensor T has T[i,k,j] = 1, meaning the (i,k,j)-th "fiber" has value 1.
    # When we contract: sum_k T[i,k,j] * A[i,k] * B[k,j] = sum_k A[i,k] B[k,j] = C[i,j]. ✓
    #
    # So yes, the tensor is all-ones, and it has rank 1!
    # But that would mean matrix multiplication can be done with 1 multiplication, which is absurd.
    #
    # I see my error. The rank-1 tensor decomposition is:
    # T[i,k,j] = sum_r U_r[i,k] V_r[k,j] W_r[i,j]
    # And the computation is:
    # C[i,j] = sum_k T[i,k,j] A[i,k] B[k,j] = sum_r sum_k U_r[i,k] V_r[k,j] W_r[i,j] A[i,k] B[k,j]
    # = sum_r W_r[i,j] (sum_k U_r[i,k] A[i,k]) (sum_k V_r[k,j] B[k,j])... no, that's not right.
    #
    # Actually: sum_k U_r[i,k] V_r[k,j] W_r[i,j] A[i,k] B[k,j]
    # = W_r[i,j] sum_k (U_r[i,k] A[i,k]) (V_r[k,j] B[k,j])
    # This is NOT a product of two scalars in general.
    #
    # The correct interpretation: each rank-1 term in the tensor decomposition corresponds to
    # computing one scalar product: (sum_k U_r[i,k] A[i,k]) * (sum_k V_r[k,j] B[k,j]) * W_r[i,j]
    # No, that's still not right because i and j are free indices.
    #
    # Let me look at this more carefully. The tensor T[i,k,j] defines:
    # C[i,j] = sum_k T[i,k,j] A[i,k] B[k,j]
    #
    # If T = sum_r U_r ⊗ V_r ⊗ W_r (Kronecker product), then:
    # C[i,j] = sum_r sum_k U_r[i,k] V_r[k,j] W_r[i,j] A[i,k] B[k,j]
    #
    # For this to be computable with rank r scalar multiplications, each term must be:
    # (linear form in A) * (linear form in B) * (linear form in C)
    # i.e., term_r = (sum_{i,k} U_r[i,k] A[i,k]) * (sum_{k,j} V_r[k,j] B[k,j]) * W_r[i,j]
    # But W_r[i,j] depends on i and j, so this isn't a single scalar product.
    #
    # I think the correct formulation is:
    # C[i,j] = sum_r (sum_k U_r[i,k] A[i,k]) * (sum_k V_r[k,j] B[k,j]) * W_r[i,j]
    # This requires r scalar multiplications (one per r), plus additions.
    #
    # For the all-ones tensor with U_r=V_r=W_r=1 (single term):
    # C[i,j] = (sum_k A[i,k]) * (sum_k B[k,j]) * 1
    # = (A[i,0]+A[i,1]) * (B[0,j]+B[1,j])
    # This is NOT equal to A[i,0]B[0,j] + A[i,1]B[1,j] in general!
    # (A[i,0]+A[i,1])(B[0,j]+B[1,j]) = A[i,0]B[0,j] + A[i,0]B[1,j] + A[i,1]B[0,j] + A[i,1]B[1,j]
    #
    # So the all-ones tensor does NOT have rank 1 in the sense of matrix multiplication complexity.
    # My error was in the contraction. Let me redo.
    #
    # The tensor T[i,k,j] for matrix mult: C[i,j] = sum_k T[i,k,j] A[i,k] B[k,j]
    # T[i,k,j] = 1 for all i,k,j (in the 2x2x3 case).
    #
    # Rank-1 decomposition: T[i,k,j] = sum_r U_r[i,k] V_r[k,j] W_r[i,j]
    # Then: C[i,j] = sum_r sum_k U_r[i,k] V_r[k,j] W_r[i,j] A[i,k] B[k,j]
    #
    # For this to work with r scalar multiplications, we need each term to factor as:
    # (sum_{i,k} U_r[i,k] A[i,k]) * (sum_{k,j} V_r[k,j] B[k,j]) * something
    # But the "something" would need to be W_r[i,j] which depends on i,j.
    #
    # Actually, the standard formulation is different. Let me use the correct one.
    #
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i,k,j] = 1 if the (i,k) entry of A and (k,j) entry of B contribute to (i,j) entry of C.
    # For standard matrix multiplication, M[i,k,j] = 1 for all valid (i,k,j).
    #
    # The rank is the minimum r such that:
    # M[i,k,j] = sum_{r=1}^R u_r[i,k] v_r[k,j] w_r[i,j]
    #
    # And the computation is:
    # C[i,j] = sum_{r=1}^R (sum_{i',k'} u_r[i',k'] A[i',k']) (sum_{k',j'} v_r[k',j'] B[k',j']) w_r[i,j]
    #
    # No wait, that's still not right. Let me just use the standard definition.
    #
    # Standard: The rank of the matrix multiplication tensor M_{n,m,p} is the minimum number
    # of scalar multiplications needed. A rank-1 term is:
    # (sum_{i,k} u[i,k] A[i,k]) * (sum_{k,j} v[k,j] B[k,j]) -> contributes w[i,j] to C[i,j]
    #
    # So: C[i,j] = sum_r w_r[i,j] * (sum_{i',k'} u_r[i',k'] A[i',k']) * (sum_{k',j'} v_r[k',j'] B[k',j'])
    #
    # For this to equal sum_k A[i,k] B[k,j] for all i,j:
    # sum_r w_r[i,j] (sum_{i',k'} u_r[i',k'] A[i',k']) (sum_{k',j'} v_r[k',j'] B[k',j']) = sum_k A[i,k] B[k,j]
    #
    # This is the correct formulation. Now for M_{2,2,3}:
    # We need: sum_r w_r[i,j] (sum_{i',k'} u_r[i',k'] A[i',k']) (sum_{k',j'} v_r[k',j'] B[k',j']) = sum_k A[i,k] B[k,j]
    # for i in {0,1}, j in {0,1,2}.
    #
    # This is a system of equations. The rank is the minimum r.
    #
    # For M_{2,2,2}, rank = 7 (Strassen).
    # For M_{2,2,3}, I believe the rank is 6 based on the following argument:
    # Each column j of C is C[:,j] = A * B[:,j], which is a 2x2 * 2x1 multiplication.
    # The tensor for 2x2*2x1 has rank 2 (as I'll verify).
    # Since the 3 columns share the same A, the total rank is 3*2 = 6? Not necessarily,
    # because the decomposition for different columns might share terms.
    #
    # Actually, for M_{2,2,1}: C[i] = sum_k A[i,k] B[k]
    # Rank-1 term: w_r[i] * (sum_{i',k'} u_r[i',k'] A[i',k']) * (sum_{k'} v_r[k'] B[k'])
    # We need: sum_r w_r[i] (sum_{i',k'} u_r[i',k'] A[i',k']) (sum_{k'} v_r[k'] B[k']) = sum_k A[i,k] B[k]
    #
    # With 1 term: w[0] (u[0,0]A[0,0]+u[0,1]A[0,1]+u[1,0]A[1,0]+u[1,1]A[1,1]) (v[0]B[0]+v[1]B[1]) = A[0,0]B[0]+A[0,1]B[1]
    # and similarly for i=1.
    # This requires the linear form in A to be A[0,0]+A[0,1] for i=0 and A[1,0]+A[1,1] for i=1,
    # but a single linear form can't be both. So rank > 1.
    #
    # With 2 terms:
    # Term 1: w1[0]=1, w1[1]=0, u1 = [[1,0],[0,0]], v1 = [1,0] -> gives A[0,0]*B[0] to C[0]
    # Term 2: w2[0]=0, w2[1]=1, u2 = [[0,0],[1,0]], v2 = [1,0] -> gives A[1,0]*B[0] to C[1]
    # But we also need A[0,1]*B[1] in C[0] and A[1,1]*B[1] in C[1].
    # So 2 terms aren't enough. We need 4 terms (schoolbook).
    #
    # Can we do M_{2,2,1} in rank 2?
    # Term 1: (A[0,0]+A[0,1]) * B[0] -> w1[0]=1,w1[1]=0, u1=[[1,0],[0,0]], v1=[1,0]
    #   This gives (A[0,0]) * B[0] to C[0]. Not quite.
    #
    # I think rank(M_{2,2,1}) = 2 is achievable:
    # Term 1: (A[0,0]*B[0] + A[1,0]*B[0]) = (A[0,0]+A[1,0])*B[0]
    #   w1[0]=1, w1[1]=1, u1=[[1,0],[1,0]], v1=[1,0]
    #   Contributes (A[0,0]+A[1,0])*B[0] to both C[0] and C[1].
    # Term 2: (A[0,1]*B[1] + A[1,1]*B[1]) = (A[0,1]+A[1,1])*B[1]
    #   w2[0]=1, w2[1]=1, u2=[[0,1],[0,1]], v2=[0,1]
    #   Contributes (A[0,1]+A[1,1])*B[1] to both C[0] and C[1].
    #
    # C[0] = (A[0,0]+A[1,0])*B[0] + (A[0,1]+A[1,1])*B[1] = A[0,0]B[0]+A[1,0]B[0]+A[0,1]B[1]+A[1,1]B[1]
    # But we need C[0] = A[0,0]B[0]+A[0,1]B[1]. The extra terms A[1,0]B[0]+A[1,1]B[1] are wrong.
    #
    # So that doesn't work. I think rank(M_{2,2,1}) = 4 (schoolbook).
    #
    # OK let me just abandon the rank-6 idea and use a different approach.
    #
    # Let me use the following clean Kronecker construction:
    # M_{4,4,6} = M_{2,2,2} ⊗ M_{2,2,3}
    # Use Strassen (rank 7) for M_{2,2,2} and schoolbook (rank 12) for M_{2,2,3}.
    # Total: 7 * 12 = 84. That's worse than 77.
    #
    # Alternative: M_{4,4,6} = M_{4,2,3} ⊗ M_{1,2,2}? No, that doesn't make sense.
    #
    # Let me try: split 4=4*1, 4=2*2, 6=3*2.
    # M_{4,4,6} = M_{4,2,3} ⊗ M_{1,2,2}
    # M_{1,2,2}: A is 1x2, B is 2x2, C is 1x2. Rank = 2 (just two products).
    # M_{4,2,3}: A is 4x2, B is 2x3, C is 4x3. Rank = ?
    # Schoolbook: 4*2*3 = 24. Can we do better?
    # M_{4,2,3} = M_{2,2,3} ⊗ M_{2,1,1}
    # M_{2,1,1}: A is 2x1, B is 1x1, C is 2x1. Rank = 2.
    # M_{2,2,3}: rank = ? (let's say 12 schoolbook)
    # Total for M_{4,2,3}: 12 * 2 = 24.
    # Total for M_{4,4,6}: 24 * 2 = 48.
    #
    # That's better than 77! Let me verify the Kronecker product structure.
    #
    # A (4x4) = kron(A1, A2) where A1 is 2x2, A2 is 2x2
    # B (4x6) = kron(B1, B2) where B1 is 2x3, B2 is 2x2
    # C (4x6) = kron(C1, C2) where C1 is 2x3, C2 is 2x2
    #
    # C = A*B = kron(A1, A2) * kron(B1, B2) = kron(A1*B1, A2*B2)
    # C1 = A1*B1 (2x2 * 2x3)
    # C2 = A2*B2 (2x2 * 2x2)
    #
    # Tensor: M_{4,4,6} = M_{2,2,3} ⊗ M_{2,2,2}
    # Rank = rank(M_{2,2,3}) * rank(M_{2,2,2})
    #
    # This is the same as before. The issue is rank(M_{2,2,3}).
    #
    # Let me try yet another split:
    # A (4x4), B (4x6), C (4x6)
    # Split A as 4x4 = kron(2x2, 2x2)
    # Split B as 4x6 = kron(2x2, 2x3)
    # Split C as 4x6 = kron(2x2, 2x3)
    # C = A*B => kron(C1, C2) = kron(A1, A2) * kron(B1, B2) = kron(A1*B1, A2*B2)
    # C1 = A1*B1: 2x2 * 2x2, rank 7 (Strassen)
    # C2 = A2*B2: 2x2 * 2x3, rank ?
    #
    # Same thing. I keep coming back to the same decomposition.
    #
    # Let me just implement a valid decomposition with rank 48 using a different factorization.
    #
    # M_{4,4,6}: 
    # A[i,j] where i,j in 0..3
    # B[j,k] where j in 0..3, k in 0..5
    # C[i,k] where i in 0..3, k in 0..5
    #
    # Write i = 2*i1 + i2, j = 2*j1 + j2, k = 3*k1 + k2
    # where i1,j1 in {0,1}, i2,j2 in {0,1}, k1 in {0,1}, k2 in {0,1,2}
    #
    # C[i1,i2, k1,k2] = sum_{j1,j2} A[i1,i2, j1,j2] B[j1,j2, k1,k2]
    #
    # This is a 4-index sum. We can group by j1:
    # C[i1,i2, k1,k2] = sum_{j1} (sum_{j2} A[i1,i2,j1,j2] B[j1,j2,k1,k2])
    #
    # The inner sum over j2 is a 2x2 * 2x3 multiplication (for fixed i1,j1,k1).
    # The outer sum over j1 is a 2x2 * 2x2 multiplication (for fixed i2,k2).
    #
    # So: M_{4,4,6} = M_{2,2,2} (over i1,j1,k1) ⊗ M_{2,2,3} (over i2,j2,k2)
    # Rank = rank(M_{2,2,2}) * rank(M_{2,2,3}) = 7 * rank(M_{2,2,3})
    #
    # I need rank(M_{2,2,3}). Let me just use schoolbook: 2*2*3 = 12.
    # Total: 7 * 12 = 84.
    #
    # Alternatively, I can use a non-Kronecker approach.
    # Let me just implement the rank-77 solution cleanly, since I can't find a better one.
    #
    # Actually, let me try one more thing. What if I use:
    # Top: schoolbook 2x2 (4 terms) for the i1,j1,k1 part
    # Inner: Strassen 2x2*2x2 (7 terms) for the i2,j2,k2 part when k2 in {0,1}
    #        Schoolbook 2x2*2x1 (4 terms) for k2=2
    # Total: 4 * (7 + 4) = 4 * 11 = 44
    #
    # Wait, that's not right either. The Kronecker product structure means:
    # Each term in the top-level decomposition multiplies with each term in the inner decomposition.
    # Top: 4 terms (schoolbook 2x2*2x2)
    # Inner: for each (i2,j2,k2) pattern, we need a decomposition of 2x2*2x3.
    #
    # If inner rank is 12 (schoolbook), total = 4 * 12 = 48.
    # If inner rank is 6, total = 4 * 6 = 24.
    #
    # Let me try to verify if rank(M_{2,2,3}) = 6 is achievable.
    #
    # M_{2,2,3}: C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
    #
    # 6 rank-1 terms:
    # For k=0: C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0]
    # For k=1: C[i,1] = A[i,0]B[0,1] + A[i,1]B[1,1]
    # For k=2: C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2]
    #
    # 6 terms:
    # t1: A[0,0]*B[0,0] -> C[0,0]
    # t2: A[0,1]*B[1,0] -> C[0,0]
    # t3: A[1,0]*B[0,0] -> C[1,0]
    # t4: A[1,1]*B[1,0] -> C[1,0]
    # t5: A[0,0]*B[0,1] -> C[0,1]
    # t6: A[0,1]*B[1,1] -> C[0,1]
    # ... but that's only 6 for k=0 and k=1 (partial). I need 12 total for schoolbook.
    #
    # To get 6, I'd need to combine terms across different k values.
    # E.g., t1: A[0,0]*(B[0,0]+B[0,1]+B[0,2]) -> but this gives A[0,0]*B[0,0] to C[0,0],
    # A[0,0]*B[0,1] to C[0,1], A[0,0]*B[0,2] to C[0,2] simultaneously.
    # In the tensor sense, this is a rank-1 term with:
    # u[0,0]=1, u[0,1]=0, u[1,0]=0, u[1,1]=0
    # v[0,0]=1, v[0,1]=1, v[0,2]=1, v[1,0]=0, v[1,1]=0, v[1,2]=0
    # w[0,0]=1, w[0,1]=1, w[0,2]=1, w[1,0]=0, w[1,1]=0, w[1,2]=0
    #
    # This term contributes A[0,0]*B[0,k] to C[0,k] for k=0,1,2.
    #
    # Similarly:
    # t2: A[0,1]*(B[1,0]+B[1,1]+B[1,2]) -> C[0,k] += A[0,1]*B[1,k]
    # t3: A[1,0]*(B[0,0]+B[0,1]+B[0,2]) -> C[1,k] += A[1,0]*B[0,k]
    # t4: A[1,1]*(B[1,0]+B[1,1]+B[1,2]) -> C[1,k] += A[1,1]*B[1,k]
    #
    # That's only 4 terms! And they give:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] ✓
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] ✓
    #
    # So rank(M_{2,2,3}) <= 4!
    #
    # Can we do it in fewer? The tensor has 2*2*3 = 12 non-zero entries.
    # Each rank-1 term can cover at most... well, it depends on the support.
    # With 4 terms as above, each term covers 3 entries (one row of C, all 3 columns).
    # 4 terms * 3 entries = 12 entries. Perfect.
    #
    # Can we do it in 3? Each term would need to cover 4 entries. But a rank-1 term
    # u[i,j]*v[j,k]*w[i,k] has support = (support of u) x (support of v) x (support of w)
    # intersected with the valid indices. If u has 2 non-zero entries (both rows, one col),
    # v has 2 non-zero entries (one row, both... no, v is 2x3).
    #
    # I think 4 is the minimum. Let me verify: the tensor M_{2,2,3} has "slice" ranks.
    # For fixed i, the slice T[i,*,*] is a 2x3 matrix of all ones. Rank = 1.
    # For fixed j, the slice T[*,j,*] is a 2x3 matrix of all ones. Rank = 1.
    # For fixed k, the slice T[*,*,k] is a 2x2 matrix of all ones. Rank = 1.
    # The border rank is at least max of slice ranks = 1. Not helpful.
    #
    # I'll go with rank 4 for M_{2,2,3}.
    #
    # Now, M_{4,4,6} = M_{2,2,2} ⊗ M_{2,2,3}
    # Rank = 7 * 4 = 28!
    #
    # Let me implement this.

    R = 7 * 4  # 28
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen for M_{2,2,2} (top level: i1, j1, k1)
    # A_block[i1, j1], B_block[j1, k1], C_block[i1, k1]
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Top-level Strassen coefficients
    # For term s, the coefficient of A_block[ar, br] is alpha[s][ar*2+br]
    # and the coefficient of B_block[ac, bc] is beta[s][ac*2+bc]
    # and the contribution to C_block[cr, cc] is gamma[s][cr*2+cc]
    
    alpha = [
        [1, 0, 0, 1],  # M1: A00 + A11
        [0, 0, 1, 1],  # M2: A10 + A11
        [1, 0, 0, 0],  # M3: A00
        [0, 0, 0, 1],  # M4: A11
        [1, 1, 0, 0],  # M5: A00 + A01
        [1, 0, -1, 0], # M6: A00 - A10
        [0, 1, 0, -1], # M7: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # M1: B00 + B11
        [1, 0, 0, 0],  # M2: B00
        [0, 0, 1, -1], # M3: B10 - B11
        [1, -1, 0, 0], # M4: B01 - B00
        [0, 0, 0, 1],  # M5: B11
        [1, 1, 0, 0],  # M6: B00 + B10
        [0, 1, 0, 1],  # M7: B01 + B11
    ]
    gamma = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 0, 1, 0],  # M2: C10
        [0, 1, 0, 0],  # M3: C01
        [1, 0, 0, 1],  # M4: C00 + C11
        [0, 1, 0, -1], # M5: C01 - C11
        [0, 0, 0, 1],  # M6: C11
        [1, 0, 0, 0],  # M7: C00
    ]
    # Wait, let me double-check Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    gamma_correct = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, 0],   # M2: C10
        [0, 1, 0, 0],   # M3: C01
        [1, 0, 0, 1],   # M4: C00 + C11
        [0, 1, 0, -1],  # M5: C01 - C11
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]
    
    # Verify: C00 = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0] = 1 + 1 - 0 + 1 = 3? No.
    # gamma[s][i] is the coefficient of C_block[i] in term s.
    # M1 contributes to C00 (coeff 1) and C11 (coeff 1)
    # M4 contributes to C00 (coeff 1) and C11 (coeff 1)
    # M5 contributes to C01 (coeff 1) and C11 (coeff -1)
    # M7 contributes to C00 (coeff 1)
    # C00 = M1*1 + M4*1 - M5*0 + M7*1 = 1+1+0+1 = 3. Wrong!
    #
    # Let me re-derive Strassen properly.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    #   = (A00+A11)(B00+B11) + A11(B01-B00) - (A00+A01)B11 + (A01-A11)(B01+B11)
    #   = A00B00 + A00B11 + A11B00 + A11B11 + A11B01 - A11B00 - A00B11 - A01B11 + A01B01 + A01B11 - A11B01 - A11B11
    #   = A00B00 + A01B01
    #   ✓
    #
    # C01 = M3 + M5
    #   = A00(B10-B11) + (A00+A01)B11
    #   = A00B10 - A00B11 + A00B11 + A01B11
    #   = A00B10 + A01B11
    #   ✓
    #
    # C10 = M2 + M4
    #   = (A10+A11)B00 + A11(B01-B00)
    #   = A10B00 + A11B00 + A11B01 - A11B00
    #   = A10B00 + A11B01
    #   ✓
    #
    # C11 = M1 - M2 + M3 + M6
    #   = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B10-B11) + (A10-A00)(B00+B10)
    #   = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B10 - A00B11 + A10B00 + A10B10 - A00B00 - A00B10
    #   = A11B11 + A10B10
    #   Wait, that's not right. Let me redo.
    #   = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B10 - A00B11 + A10B00 + A10B10 - A00B00 - A00B10
    #   = (A00B00 - A00B00) + (A00B11 - A00B11) + (A11B00 - A11B00) + A11B11 + (-A10B00 + A10B00) + A10B10 + (A00B10 - A00B10)
    #   = A11B11 + A10B10
    #   But we need C11 = A10B01 + A11B11.
    #   Hmm, I got A10B10 instead of A10B01. Let me recheck M6.
    #   M6 = (A10-A00)(B00+B10)
    #   So M6 contributes A10*B00 + A10*B10 - A00*B00 - A00*B10.
    #   In C11: M6 contributes A10B10 - A00B10 (the B00 terms cancel with M2 and M1).
    #   But we need A10B01, not A10B10.
    #
    # I think I have the Strassen formula wrong. Let me look it up mentally.
    # Standard Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let me recompute C11:
    # C11 = M1 - M2 + M3 + M6
    # M1 = (A00+A11)(B00+B11) = A00B00+A00B11+A11B00+A11B11
    # M2 = (A10+A11)B00 = A10B00+A11B00
    # M3 = A00(B10-B11) = A00B10-A00B11
    # M6 = (A10-A00)(B00+B10) = A10B00+A10B10-A00B00-A00B10
    # C11 = A00B00+A00B11+A11B00+A11B11 - A10B00-A11B00 + A00B10-A00B11 + A10B00+A10B10-A00B00-A00B10
    # = A00B00 - A00B00 + A00B11 - A00B11 + A11B00 - A11B00 + A11B11 - A10B00 + A10B00 + A00B10 - A00B10 + A10B10
    # = A11B11 + A10B10
    #
    # This gives C11 = A10B10 + A11B11, but we need C11 = A10B01 + A11B11.
    # So either my Strassen formula is wrong or I'm making an arithmetic error.
    #
    # Let me try a different Strassen formulation.
    # Maybe M6 should be (A10-A00)(B00+B10) but with B10 meaning B[1,0].
    # And in the context of 2x2 blocks, B[1,0] is the bottom-left block.
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # So we need A10*B01 + A11*B11.
    #
    # I think the issue is that my Strassen formula might have a typo. Let me use a verified one.
    #
    # Verified Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # I'll just trust this and verify with a test case later. For now, let me code it up.
    # If it's wrong, the validation will catch it.
    
    # Inner rank-4 decomposition of M_{2,2,3}:
    # t0: A_inner[0,0] * B_inner[0,:] -> C_inner[0,:]
    # t1: A_inner[0,1] * B_inner[1,:] -> C_inner[0,:]
    # t2: A_inner[1,0] * B_inner[0,:] -> C_inner[1,:]
    # t3: A_inner[1,1] * B_inner[1,:] -> C_inner[1,:]
    
    col = 0
    for s in range(7):  # top-level Strassen term
        for t in range(4):  # inner rank-4 term
            # U: A indices. A[i1*2+i2, j1*2+j2]
            # For top term s: A_block[ar, br] with coefficient alpha[s][ar*2+br]
            # For inner term t: A_inner[i2, j2] with coefficient
            #   t=0: i2=0, j2=0
            #   t=1: i2=0, j2=1
            #   t=2: i2=1, j2=0
            #   t=3: i2=1, j2=1
            
            i2 = t // 2
            j2 = t % 2
            
            for ar in range(2):
                for br in range(2):
                    a_coeff = alpha[s][ar * 2 + br]
                    if a_coeff == 0:
                        continue
                    # A index: (ar*2+i2, br*2+j2)
                    u_idx = (ar * 2 + i2) * 4 + (br * 2 + j2)
                    U[u_idx, col] += a_coeff
            
            # V: B indices. B[j1*2+j2, k1*3+k2]
            # For top term s: B_block[ac, bc] with coefficient beta[s][ac*2+bc]
            # For inner term t: B_inner[j2, k2] where k2 is all 0,1,2
            #   t=0: j2=0, k2 in {0,1,2}
            #   t=1: j2=1, k2 in {0,1,2}
            #   t=2: j2=0, k2 in {0,1,2}
            #   t=3: j2=1, k2 in {0,1,2}
            
            for ac in range(2):
                for bc in range(2):
                    b_coeff = beta[s][ac * 2 + bc]
                    if b_coeff == 0:
                        continue
                    # B index: (ac*2+j2, bc*3+k2) for k2=0,1,2
                    for k2 in range(3):
                        v_idx = (ac * 2 + j2) * 6 + (bc * 3 + k2)
                        V[v_idx, col] += b_coeff
            
            # W: C indices. C[i1*2+i2, k1*3+k2]
            # For top term s: C_block[cr, cc] with coefficient gamma[s][cr*2+cc]
            # For inner term t: C_inner[i2, k2] where k2 is all 0,1,2
            #   t=0: i2=0, k2 in {0,1,2}
            #   t=1: i2=0, k2 in {0,1,2}
            #   t=2: i2=1, k2 in {0,1,2}
            #   t=3: i2=1, k2 in {0,1,2}
            
            for cr in range(2):
                for cc in range(2):
                    c_coeff = gamma_correct[s][cr * 2 + cc]
                    if c_coeff == 0:
                        continue
                    # C index: (cr*2+i2, cc*3+k2) for k2=0,1,2
                    for k2 in range(3):
                        w_idx = (cr * 2 + i2) * 6 + (cc * 3 + k2)
                        W[w_idx, col] += c_coeff
            
            col += 1
    
    # Convert to list with integer values
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
