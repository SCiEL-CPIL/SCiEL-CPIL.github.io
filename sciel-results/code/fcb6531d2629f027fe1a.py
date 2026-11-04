import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Winograd Generalized Algorithm (WGA) approach:
    # Decompose 2x4x6 as (2x2x2) ⊗ (1x2x3) using Kronecker product structure
    # 4 = 2*2, 6 = 2*3
    # The matmul tensor for (n1*n2) x (m1*m2) x (p1*p2) is the Kronecker product
    # of the tensors for n1xm1xp1 and n2xm2xp2
    
    # For our case: (2*1) x (2*2) x (2*3)
    # Factor 1: 2x2x2 -> rank 7 (Strassen)
    # Factor 2: 1x2x3 -> rank 6 (schoolbook, since n=1)
    # Total rank: 7 * 6 = 42... that's the same.
    
    # Actually let me think more carefully. The dimensions are:
    # A is 2x4, B is 4x6, C is 2x6
    # n=2, m=4, p=6
    # We can write m = m1*m2 = 2*2, p = p1*p2 = 2*3
    # Then C[i,k] = sum_j A[i,j]*B[j,k]
    # With j = j1*m2 + j2, k = k1*p2 + k2
    # C[i, k1*p2+k2] = sum_{j1,j2} A[i, j1*m2+j2] * B[j1*m2+j2, k1*p2+k2]
    # = sum_{j1,j2} (sum over the "inner" products)
    
    # The tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # = T[i*(m1*m2)+(j1*m2+j2), (j1*m2+j2)*(p1*p2)+(k1*p2+k2), (k1*p2+k2)*n+i]
    # = T[i*m1*m2+j1*m2+j2, j1*m2*p1*p2+j2*p1*p2+k1*p2+k2, k1*p2*n+k2*n+i]
    
    # This is a Kronecker product of two tensors:
    # T1 for (n=2, m1=2, p1=2): T1[i,j1,k1] with indices a1=i*2+j1, b1=j1*2+k1, c1=k1*2+i
    # T2 for (n=1, m2=2, p2=3): T2[i2,j2,k2] with indices a2=i2*2+j2, b2=j2*3+k2, c2=k2*1+i2
    # Wait, n=2 for the first factor, n=1 for the second? No...
    
    # Let me reconsider. The full tensor has:
    # a = i*m + j = i*(m1*m2) + j1*m2 + j2
    # b = j*p + k = (j1*m2+j2)*(p1*p2) + k1*p2 + k2
    # c = k*n + i = (k1*p2+k2)*n + i
    
    # The Kronecker product structure: if we define
    # A1[i,j1] = A[i, j1*m2 + j2] for each j2... no that's not quite right.
    
    # Actually the correct WGA decomposition:
    # Write A as a 2D array of 1D blocks, B similarly, then C is computed
    # via tensor product of the two smaller multiplications.
    
    # For 2x4x6: think of it as (2x2) x (2x2) x (2x3) in the sense that
    # the "outer" multiplication is 2x2x2 (rank 7 via Strassen)
    # and the "inner" multiplication is 1x2x3 (rank 6 via schoolbook)
    # But wait, the "inner" has n=1? That doesn't work directly.
    
    # Let me just try the direct Kronecker product approach:
    # The matmul tensor for (n1*n2) x (m1*m2) x (p1*p2) 
    # = matmul_tensor(n1,m1,p1) ⊗ matmul_tensor(n2,m2,p2)
    # in the appropriate index ordering.
    
    # For n=2, m=4, p=6: try n1=2, n2=1, m1=2, m2=2, p1=2, p2=3
    # T_full = T(2,2,2) ⊗ T(1,2,3)
    # rank = rank(T(2,2,2)) * rank(T(1,2,3)) = 7 * 6 = 42
    
    # Hmm, that gives 42 again. Let me try other factorizations.
    # n=2: only 2=2*1 or 2=1*2
    # m=4: 4=4*1, 1*4, 2*2
    # p=6: 6=6*1, 1*6, 2*3, 3*2
    
    # The key insight: we need to find a factorization where the total rank is minimized.
    # rank(T(n,m,p)) for the Kronecker product = rank1 * rank2
    
    # T(2,2,2) has rank 7 (Strassen)
    # T(1,2,3) has rank 6 (schoolbook since n=1)
    # T(2,4,2) - what's its rank? 
    # T(2,2,3) - what's its rank?
    
    # Let me try a different approach: directly construct the WGA decomposition
    # by treating the 2x4x6 problem as a 2x2x2 "super" multiplication where
    # each "element" is itself a small matrix product.
    
    # Actually, the cleanest WGA for 2x4x6:
    # View A (2x4) as a 2x2 block matrix with 1x2 blocks
    # View B (4x6) as a 2x2 block matrix with 2x3 blocks  
    # Then C (2x6) = A*B is a 2x2 block matrix with 1x3 blocks
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # With blocks: C_block[i1,k1] = sum_j1 A_block[i1,j1] * B_block[j1,k1]
    # where A_block[i1,j1] is 1x2, B_block[j1,k1] is 2x3, C_block[i1,k1] is 1x3
    
    # The "outer" multiplication is 2x2x2 (rank 7)
    # The "inner" multiplication is 1x2x3 (rank 6)
    # Total rank = 7 * 6 = 42
    
    # That's still 42. The WGA with this factorization doesn't improve over naive Strassen tiling.
    
    # Let me try yet another approach: use the fact that we can share multiplications
    # across the full 2x4x6 problem more cleverly.
    
    # Alternative: try to find a rank < 42 decomposition by searching for
    # linear dependencies in a larger Strassen-like construction.
    
    # Construct the standard Strassen tiling: 1 block of 2x2x2 (7 mults) 
    # for each of the 2x2=4 positions in the m,p grid... wait.
    
    # m=4, p=6: we can tile as 2x3 blocks of size 2x2
    # That's 2*3 = 6 Strassen blocks, each rank 7, total rank 42.
    # That's what the current code does.
    
    # To do better, we need cross-block sharing. Let me try a different tiling:
    # Use 2x2 Strassen for the 2x2 subproblems but with overlapping windows.
    
    # Or: try to find a direct low-rank decomposition via numerical methods.
    
    # Let me try: build the full tensor, then use iterative rank minimization.
    
    # First, build the target tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a rank-r decomposition by solving a linear system
    # T[a,b,c] = sum_{r=1}^{R} U[a,r] * V[b,r] * W[c,r]
    
    # Strategy: use alternating minimization or direct linear algebra
    # For fixed U,V: W is determined by linear equations
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For each c: T[:,:,c] = sum_r (U[:,r]*V[:,r]) * W[c,r]
    # So T[:,:,c] must be in the span of the outer products U[:,r]*V[:,r]
    
    # Let me try a different structural approach:
    # Use the "border rank" or try to find a decomposition by fixing one matrix
    # and solving for the others.
    
    # Actually, let me try a completely different approach:
    # The known lower bound for 2x4x6 matmul rank.
    # For 2x2x2: rank 7
    # For 2x3x3: rank 14 (2*7)
    # For 2x4x4: rank 28 (2*14) or 4*7=28
    # For 2x4x6: could be 7*6=42, or maybe less with clever sharing.
    
    # Let me try to construct a rank-28 decomposition by viewing 2x4x6 as:
    # (2x2x2) ⊗ (1x2x3) but with a non-trivial 1x2x3 decomposition.
    # T(1,2,3): T[i,j,k] with i in {0}, j in {0,1}, k in {0,1,2}
    # T[0, j, k] = 1 for all j,k. This is just a rank-6 tensor (schoolbook).
    # But wait: T(1,2,3) has only 6 nonzero entries, all = 1.
    # Can we decompose it with rank < 6?
    # T[0,j,k] = sum_r U[0,r]*V[j,r]*W[k,r]
    # Since U has only 1 row, U[0,r] is a scalar.
    # T[0,j,k] = sum_r u_r * v_{j,r} * w_{k,r}
    # This is a matrix M[j,k] = sum_r u_r * v_{j,r} * w_{k,r} = (V * diag(u)) @ W^T
    # So M = V @ diag(u) @ W^T, which has rank <= number of nonzero u_r.
    # M is a 2x3 all-ones matrix, which has rank 1!
    # So we can do it with rank 1: u_1=1, v_{j,1}=1, w_{k,1}=1.
    # T(1,2,3) has rank 1!
    
    # So T(2,4,6) = T(2,2,2) ⊗ T(1,2,3) has rank 7 * 1 = 7???
    # That can't be right. Let me verify.
    
    # The issue: the Kronecker product structure requires the index ordering
    # to match. Let me be very careful.
    
    # Full tensor: T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # n=2, m=4, p=6
    # a = 4*i + j, j = 2*j1 + j2 (j1 in {0,1}, j2 in {0,1})
    # b = (2*j1+j2)*6 + k = 12*j1 + 6*j2 + k, k = 2*k1 + k2 (k1 in {0,1,2}... wait p=6, p2=3, so k = 3*k1 + k2? No.)
    
    # Let me redo: m = m1*m2 = 2*2, so j = j1*2 + j2, j1 in {0,1}, j2 in {0,1}
    # p = p1*p2 = 2*3, so k = k1*3 + k2, k1 in {0,1}, k2 in {0,1,2}
    
    # a = i*4 + j = i*4 + j1*2 + j2
    # b = j*6 + k = (j1*2+j2)*6 + k1*3 + k2 = j1*12 + j2*6 + k1*3 + k2
    # c = k*2 + i = (k1*3+k2)*2 + i = k1*6 + k2*2 + i
    
    # For the Kronecker product T = T1 ⊗ T2:
    # T1[i,j1,k1] with a1 = i*2+j1, b1 = j1*2+k1, c1 = k1*2+i  (2x2x2 matmul)
    # T2[?,j2,k2] with a2 = ?*2+j2, b2 = j2*3+k2, c2 = k2*?+?  (1x2x3 matmul)
    
    # The problem: c = k1*6 + k2*2 + i, but for T1: c1 = k1*2+i, for T2: c2 = k2*1+i2
    # c1 * p2 + c2 = (k1*2+i)*3 + (k2*1+i2) = k1*6 + i*3 + k2 + i2
    # But actual c = k1*6 + k2*2 + i. These don't match!
    
    # The index ordering for the Kronecker product needs to be:
    # a = a1*m2 + a2 = (i*m1+j1)*2 + (i2*m2+j2) ... this is getting complicated.
    
    # The correct Kronecker product for matmul tensors:
    # If we have matmul(n1,m1,p1) and matmul(n2,m2,p2), the product tensor
    # matmul(n1*n2, m1*m2, p1*p2) is NOT simply the Kronecker product in general.
    # It IS the Kronecker product if we use the right index ordering.
    
    # The matmul tensor T[i,j,k] = 1 if the indices are compatible.
    # For the Kronecker product: T_full[i1*i2, j1*j2, k1*k2] 
    # = T1[i1,j1,k1] * T2[i2,j2,k2]
    # where i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    
    # Let me verify: with i = i1*2+i2, j = j1*2+j2, k = k1*3+k2
    # a = i*4+j = (i1*2+i2)*4 + j1*2+j2 = i1*8 + i2*4 + j1*2 + j2
    # b = j*6+k = (j1*2+j2)*6 + k1*3+k2 = j1*12 + j2*6 + k1*3 + k2
    # c = k*2+i = (k1*3+k2)*2 + i1*2+i2 = k1*6 + k2*2 + i1*2 + i2
    
    # T1[i1,j1,k1]: a1=i1*2+j1, b1=j1*2+k1, c1=k1*2+i1
    # T2[i2,j2,k2]: a2=i2*2+j2, b2=j2*3+k2, c2=k2*1+i2
    
    # a1*m2 + a2 = (i1*2+j1)*2 + (i2*2+j2) = i1*4+j1*2+i2*2+j2. But a = i1*8+i2*4+j1*2+j2. Doesn't match.
    
    # The correct ordering: a = a1*(m2) + a2 only works if a1 uses the "slow" index.
    # a1 = i1*m1 + j1 = i1*2 + j1, this ranges over 0..3
    # a2 = i2*m2 + j2 = i2*2 + j2, this ranges over 0..3
    # a = a1*m2 + a2 = (i1*2+j1)*2 + (i2*2+j2) = i1*4+j1*2+i2*2+j2
    # But actual a = i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8+i2*4+j1*2+j2
    # These are different orderings!
    
    # So the simple Kronecker product doesn't directly apply with this index ordering.
    # We need to use a permutation of the indices.
    
    # This is getting complex. Let me try a more direct approach:
    # Just try to find a low-rank decomposition numerically.
    
    # Approach: use the "tensor train" or "CP decomposition" approach.
    # Or: fix the structure and solve linear equations.
    
    # Let me try rank 28: use the Strassen 2x2x2 core (rank 7) and a 2x2x3 "convolution" (rank 4).
    # 7 * 4 = 28.
    
    # For the 2x2x3 part: matmul(2,2,3)
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2}
    # a = i*2+j, b = j*3+k, c = k*2+i
    # Can we decompose this with rank 4?
    # T has 12 nonzero entries. The matrix unfolding T[:,:,c] for each c:
    # c=0: k=0,i=0: T[a,b,0] nonzero when k=0,i=0, so a=i*2+j=0*2+j=j, b=j*3+0=3j
    #   T[0,0,0]=1, T[1,3,0]=1 -> 2x6 matrix with 2 nonzeros
    # c=1: k=0,i=1 or k=1,i=0... wait c=k*2+i
    #   c=0: k=0,i=0
    #   c=1: k=0,i=1
    #   c=2: k=1,i=0
    #   c=3: k=1,i=1
    #   c=4: k=2,i=0
    #   c=5: k=2,i=1
    
    # For c=0 (k=0,i=0): a=j, b=3j. T[0,0]=1, T[1,3]=1
    # For c=1 (k=0,i=1): a=j, b=3j. T[0,0]=1, T[1,3]=1
    # For c=2 (k=1,i=0): a=j, b=3j+1. T[0,1]=1, T[1,4]=1
    # For c=3 (k=1,i=1): a=j, b=3j+1. T[0,1]=1, T[1,4]=1
    # For c=4 (k=2,i=0): a=j, b=3j+2. T[0,2]=1, T[1,5]=1
    # For c=5 (k=2,i=1): a=j, b=3j+2. T[0,2]=1, T[1,5]=1
    
    # So T[:,:,0] = T[:,:,1] and T[:,:,2] = T[:,:,3] and T[:,:,4] = T[:,:,5]
    # Each is a 4x6 matrix with two 1s.
    # The span of {T[:,:,c]} has dimension 3 (three distinct matrices).
    # So the "c-unfolding" has rank 3.
    
    # For a rank-R decomposition: T[:,:,c] = sum_r U[:,r]*V[:,r] * W[c,r]
    # The matrices U[:,r]*V[:,r] must span the space of T[:,:,c] matrices.
    # Since there are only 3 distinct matrices (each with 2 nonzeros), we need
    # at least... let me think.
    
    # T[:,:,0] = E[0,0] + E[1,3]
    # T[:,:,2] = E[0,1] + E[1,4]
    # T[:,:,4] = E[0,2] + E[1,5]
    
    # Can we write these as sums of rank-1 matrices with coefficients?
    # E[0,0]+E[1,3] = u0*v0^T * w0 + u1*v1^T * w1 + ...
    # This is getting complicated. Let me just try rank 4 for the 2x2x3 part.
    
    # Actually, let me try a completely different strategy:
    # Use numerical optimization to find a low-rank CP decomposition.
    
    # Or better: use the known result that matmul(2,4,6) has rank at most 42
    # (from Strassen tiling) and try to find if it can be lower.
    
    # Let me try a direct approach: assume rank R, and use the fact that
    # the tensor is sparse to set up a system of equations.
    
    # For rank R: T[a,b,c] = sum_{r=1}^{R} U[a,r]*V[b,r]*W[c,r]
    # We have 48 equations (one for each nonzero entry of T) and 3*R*(R+something) unknowns.
    # Actually U has nm*R = 8R entries, V has mp*R = 24R entries, W has pn*R = 12R entries.
    # Total: 44R unknowns, 48 equations. For R >= 2, we have more unknowns than equations.
    # But the equations are nonlinear (products of U,V,W entries).
    
    # Let me try a specific construction for rank 28:
    # Use 7 Strassen multiplications for the 2x2x2 "outer" product,
    # and for each, a 2x2x3 "inner" product with rank 4.
    
    # The 2x2x3 matmul tensor:
    # T2[i,j,k]: a=i*2+j, b=j*3+k, c=k*2+i
    # Nonzero: T2[0,0,0]=1, T2[0,3,1]=1, T2[1,1,2]=1, T2[1,4,3]=1, T2[0,2,4]=1, T2[0,5,5]=1
    # Wait let me recompute:
    # i=0,j=0,k=0: a=0, b=0, c=0 -> T[0,0,0]=1
    # i=0,j=0,k=1: a=0, b=1, c=2 -> T[0,1,2]=1
    # i=0,j=0,k=2: a=0, b=2, c=4 -> T[0,2,4]=1
    # i=0,j=1,k=0: a=1, b=3, c=0 -> T[1,3,0]=1
    # i=0,j=1,k=1: a=1, b=4, c=2 -> T[1,4,2]=1
    # i=0,j=1,k=2: a=1, b=5, c=4 -> T[1,5,4]=1
    # i=1,j=0,k=0: a=2, b=0, c=1 -> T[2,0,1]=1
    # i=1,j=0,k=1: a=2, b=1, c=3 -> T[2,1,3]=1
    # i=1,j=0,k=2: a=2, b=2, c=5 -> T[2,2,5]=1
    # i=1,j=1,k=0: a=3, b=3, c=1 -> T[3,3,1]=1
    # i=1,j=1,k=1: a=3, b=4, c=3 -> T[3,4,3]=1
    # i=1,j=1,k=2: a=3, b=5, c=5 -> T[3,5,5]=1
    
    # 12 nonzeros. Can we decompose with rank 4?
    # T2[a,b,c] = sum_{r=1}^{4} U2[a,r]*V2[b,r]*W2[c,r]
    # 12 equations, 4*(4+6+6)=56 unknowns. Should be feasible.
    
    # Let me try to solve this. I'll use a specific ansatz.
    # Notice the structure: for fixed (i,k), the nonzero entries are at a=i*2+j, b=j*3+k
    # for j=0,1. So for each (i,k), we have two entries.
    
    # Let me try: U2 has 4x4, V2 has 6x4, W2 has 6x4.
    # I'll try to find it numerically.
    
    # Actually, let me just try the full problem directly with a rank guess.
    
    # Strategy: try ranks from 24 upward, and for each rank, try to find
    # a decomposition using numerical methods.
    
    # For a given rank R, we need to find U (8xR), V (24xR), W (12xR) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c.
    
    # This is a system of 8*24*12 = 2304 equations (but only 48 are nonzero).
    # The nonzero equations are:
    # For each (i,j,k): U[i*4+j, r] * V[j*6+k, r] * W[k*2+i, r] summed over r = 1
    
    # Let me use a randomized approach: initialize U,V,W randomly, then
    # use alternating least squares or gradient descent.
    
    # But this might not converge to the global minimum.
    
    # Alternative: use the "border rank" approach or try specific structured decompositions.
    
    # Let me try a specific construction for rank 28:
    # The idea: use 4 "Strassen-like" multiplications for the 2x2 blocks in the
    # i-j plane, and handle the k dimension separately.
    
    # Actually, let me try the simplest possible improvement: 
    # The current rank-42 decomposition uses 6 independent Strassen blocks.
    # Can we share any multiplications between blocks?
    
    # Block (i1,j1,k1) uses A[i1*2:..., j1*2:...] and B[j1*2:..., k1*2:...]
    # Adjacent blocks share rows/columns of A and B.
    
    # For example, block (0,0,0) and block (0,0,1) share the same A rows.
    # Block (0,0,0) uses A[0,0], A[0,1], A[1,0], A[1,1] (for the 2x2 submatrix)
    # Block (0,0,1) uses the same A submatrix but different B columns.
    
    # In Strassen, M1 = (A00+A11)(B00+B11). For block (0,0,0):
    # M1_000 = (A[0,0]+A[1,1]) * (B[0,0]+B[2,0])  [using j=0,1 and k=0,1]
    # For block (0,0,1):
    # M1_001 = (A[0,0]+A[1,1]) * (B[0,2]+B[2,2])  [using j=0,1 and k=2,3]
    
    # These share the factor (A[0,0]+A[1,1])! So we can compute it once.
    # But in the tensor decomposition, each "multiplication" is a separate term.
    # The sharing would mean that two terms have the same U vector but different V vectors.
    # That's already allowed in the decomposition!
    
    # The issue is: in the rank-42 decomposition, each Strassen block contributes
    # 7 independent terms. If two blocks share a U factor, that doesn't reduce the rank
    # because the V factors are different.
    
    # To reduce rank, we need actual linear dependencies among the 42 term tensors.
    
    # Let me check: are the 42 term tensors linearly independent?
    # Each term is U[:,r] ⊗ V[:,r] ⊗ W[:,r].
    # Two terms are linearly dependent if one is a scalar multiple of the other,
    # which requires U[:,r1] = c*U[:,r2], V[:,r1] = c*V[:,r2], W[:,r1] = c*W[:,r2].
    # Since the Strassen blocks use different A,B submatrices, this is unlikely.
    
    # But there could be more complex linear dependencies involving 3+ terms.
    
    # Let me just check numerically if the 42 terms are independent.
    
    # Build the 42 term vectors
    U42 = np.zeros((nm, 42), dtype=float)
    V42 = np.zeros((mp, 42), dtype=float)
    W42 = np.zeros((pn, 42), dtype=float)
    
    def add_strassen_block(i_start, j_start, k_start, offset, U_mat, V_mat, W_mat):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)

        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)

        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)

        # M1: (A00+A11)(B00+B11)
        U_mat[a00, offset] += 1.0
        U_mat[a11, offset] += 1.0
        V_mat[b00, offset] += 1.0
        V_mat[b11, offset] += 1.0
        W_mat[w00, offset] += 1.0
        W_mat[w11, offset] += 1.0

        # M2: (A10+A11)B00
        U_mat[a10, offset + 1] += 1.0
        U_mat[a11, offset + 1] += 1.0
        V_mat[b00, offset + 1] += 1.0
        W_mat[w10, offset + 1] += 1.0
        W_mat[w11, offset + 1] -= 1.0

        # M3: A00(B01-B11)
        U_mat[a00, offset + 2] += 1.0
        V_mat[b01, offset + 2] += 1.0
        V_mat[b11, offset + 2] -= 1.0
        W_mat[w01, offset + 2] += 1.0
        W_mat[w11, offset + 2] += 1.0

        # M4: A11(B10-B00)
        U_mat[a11, offset + 3] += 1.0
        V_mat[b10, offset + 3] += 1.0
        V_mat[b00, offset + 3] -= 1.0
        W_mat[w00, offset + 3] += 1.0
        W_mat[w10, offset + 3] += 1.0

        # M5: (A00+A01)B11
        U_mat[a00, offset + 4] += 1.0
        U_mat[a01, offset + 4] += 1.0
        V_mat[b11, offset + 4] += 1.0
        W_mat[w00, offset + 4] -= 1.0
        W_mat[w01, offset + 4] += 1.0

        # M6: (A10-A00)(B00+B01)
        U_mat[a10, offset + 5] += 1.0
        U_mat[a00, offset + 5] -= 1.0
        V_mat[b00, offset + 5] += 1.0
        V_mat[b01, offset + 5] += 1.0
        W_mat[w11, offset + 5] += 1.0

        # M7: (A01-A11)(B10+B11)
        U_mat[a01, offset + 6] += 1.0
        U_mat[a11, offset + 6] -= 1.0
        V_mat[b10, offset + 6] += 1.0
        V_mat[b11, offset + 6] += 1.0
        W_mat[w00, offset + 6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset, U42, V42, W42)
                offset += 7

    # Check linear independence of the 42 terms
    terms = []
    for r in range(42):
        term_vec = np.zeros(nm * mp * pn)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    term_vec[a * mp * pn + b * pn + c] = U42[a, r] * V42[b, r] * W42[c, r]
        terms.append(term_vec)
    
    terms_mat = np.column_stack(terms)
    U_rank, s_rank, Vt_rank = np.linalg.svd(terms_mat, full_matrices=False)
    tol_rank = 1e-10 * s_rank[0] if len(s_rank) > 0 else 1e-10
    numerical_rank = int(np.sum(s_rank > tol_rank))
    
    # Now try to find a minimal set of terms that still spans the tensor
    # If numerical_rank < 42, we can potentially reduce the rank.
    
    best_rank = 42
    best_U = U42
    best_V = V42
    best_W = W42
    
    if numerical_rank < 42:
        # Try to find a subset of terms of size numerical_rank that spans the tensor
        # Use greedy selection
        selected = []
        remaining = list(range(42))
        
        while len(selected) < numerical_rank and remaining:
            if not selected:
                sel_idx = remaining[0]
            else:
                sel_terms = np.column_stack([terms_mat[:, s] for s in selected])
                best_sel = -1
                best_residual = float('inf')
                for r in remaining:
                    x = np.linalg.lstsq(sel_terms, terms_mat[:, r], rcond=None)[0]
                    recon = sel_terms @ x
                    residual = np.linalg.norm(terms_mat[:, r] - recon)
                    if residual < best_residual:
                        best_residual = residual
                        best_sel = r
                sel_idx = best_sel
            
            selected.append(sel_idx)
            remaining.remove(sel_idx)
        
        new_rank = len(selected)
        U_new = np.zeros((nm, new_rank), dtype=float)
        V_new = np.zeros((mp, new_rank), dtype=float)
        W_new = np.zeros((pn, new_rank), dtype=float)
        
        for new_r, old_r in enumerate(selected):
            U_new[:, new_r] = U42[:, old_r]
            V_new[:, new_r] = V42[:, old_r]
            W_new[:, new_r] = W42[:, old_r]
        
        # Verify
        def verify_decomp(U_mat, V_mat, W_mat, rank_val, tol=1e-8):
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        s = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                        if abs(s - 1.0) > tol:
                            return False
            return True
        
        if verify_decomp(U_new, V_new, W_new, new_rank):
            best_rank = new_rank
            best_U = U_new
            best_V = V_new
            best_W = W_new
    
    # Verify final
    def verify_final(U_mat, V_mat, W_mat, rank_val, tol=1e-8):
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify_final(best_U, best_V, best_W, best_rank):
        # Fall back to schoolbook rank 48
        best_rank = n * m * p
        best_U = np.zeros((nm, best_rank), dtype=float)
        best_V = np.zeros((mp, best_rank), dtype=float)
        best_W = np.zeros((pn, best_rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    best_U[i * m + j, r] = 1
                    best_V[j * p + k, r] = 1
                    best_W[k * n + i, r] = 1

    if time.time() > deadline - 5:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
