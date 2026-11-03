"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # The multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid (i,j,k)
    
    # Strategy: Use the matrix flattening trick.
    # Reshape the tensor into a matrix of size (n*p) x (m*p) where:
    # M[(i,k), (j, l)] = T[i,j,k] if l == k else 0
    # But this gives a matrix with many zeros. Instead, use the standard approach:
    # The tensor can be viewed as a linear map. For each fixed j, we have a matrix A_j of size n x p
    # where A_j[i,k] = T[i,j,k] = 1 if the indices are valid.
    
    # Actually, the cleanest approach: set up the linear system directly.
    # For rank r, we have r columns. For each column c, U[:,c], V[:,c], W[:,c] are vectors.
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of n*m*p = 36 equations in r*(n*m + m*p + p*n) = r*36 unknowns.
    # But it's nonlinear (trilinear). Instead, fix two matrices and solve for the third.
    
    # Better approach: Use the fact that we can parameterize. Let's fix V and W to be sparse,
    # then solve for U linearly.
    
    # Construction for rank 20:
    # For each k in 0..5, use 3 terms. That's 18 terms. Plus 2 extra terms for efficiency.
    # Actually, let me just construct a valid rank-20 decomposition explicitly.
    
    # Key insight: For fixed k, the sub-tensor is T[i,j,k] = 1 for all i,j.
    # This is the outer product of [1,1] (n-vector) and [1,1,1] (m-vector), rank 1 as a matrix.
    # But in the tensor decomposition, each term couples i, j, k together.
    
    # Let me try: for each k (6 values), use 3 terms that each handle one j-value but both i-values.
    # Term for (j,k): U[i*m+j, c] = 1 for both i, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for both i.
    # But then sum_c would give 3 for each (i,k), not 1. We need coefficients.
    
    # Use U[i*m+j,c] = 1/2, V[j*p+k,c] = 1, W[k*n+i,c] = 2? No, that gives 1 per term, 3 total.
    
    # I need a different approach. Let me use 2 terms per (j,k) but share across j.
    
    # Actually, let me just solve it numerically. I'll use a greedy approach:
    # Start with rank 20, set up the trilinear system, and solve iteratively.
    
    # Simpler: construct explicitly using a known pattern.
    
    # For (2,3,6), use rank 18: for each of 6 k-values, use 3 terms.
    # For term (j,k) where j in 0,1,2 and k in 0..5:
    #   U[i*m+j, c] = 1 for i=0,1
    #   V[j*p+k, c] = 1  
    #   W[k*n+i, c] = 1 for i=0,1
    # This gives sum over j of 1*1*1 = 3 for each (i,k). Need to divide by 3.
    
    # Use U[i*m+j,c] = 1/3, V[j*p+k,c] = 1, W[k*n+i,c] = 3? 
    # Then product = (1/3)*1*3 = 1 per term, 3 terms sum to 3. Still wrong.
    
    # I need exactly 1. Use U=1, V=1, W=1/3? Then each term gives 1/3, 3 terms give 1. Yes!
    
    rank = m * p  # 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = Fraction(1, 3)
            col += 1
    
    # Verify: for fixed (i,j,k), sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Only the term with matching (j,k) and this i contributes: 1*1*(1/3) = 1/3. 
    # But we sum over all j' and the other i'... wait, no. For fixed (i,j,k), only one term matches.
    # So the sum is 1/3, not 1. This is wrong.
    
    # I need the sum over c to be 1 for each (i,j,k). Let me reconsider.
    
    # Reset and use a cleaner construction.
    
    # For each (i,j,k) triple, I need exactly one term that contributes 1.
    # That's 36 terms minimum if they're all separate. To reduce, I need terms that contribute
    # to multiple (i,j,k) triples simultaneously.
    
    # A term with U[a], V[b], W[c] contributes to all (i,j,k) where:
    #   i*m+j = a, j*p+k = b, k*n+i = c
    # For this to be consistent: j from a, k from b, i from c must agree.
    
    # So each term corresponds to at most one (i,j,k) triple! The rank must be at least 36?
    # No wait, that's not right. The indices in U, V, W are independent per term.
    
    # Let me re-examine: U has indices (i,j), V has indices (j,k), W has indices (k,i).
    # For a given term c, U[:,c] is a vector over (i,j), V[:,c] over (j,k), W[:,c] over (k,i).
    # The contribution to T[i,j,k] is U[i,j,c]*V[j,k,c]*W[k,i,c].
    # For a fixed c, this is nonzero for all (i,j,k) where all three are nonzero.
    # If U[i,j,c]=1 for all i,j, V[j,k,c]=1 for all j,k, W[k,i,c]=1 for all k,i,
    # then the contribution is 1 for all 36 triples. So one term covers all 36!
    # But then we'd need coefficient 1/36. Let's check:
    
    # Term: U[i*m+j,c] = 1/36 for all, V[j*p+k,c] = 1 for all, W[k*n+i,c] = 36 for all.
    # Product = (1/36)*1*36 = 1 for all (i,j,k). Sum over c (just one term) = 1. 
    # But wait, we need the sum to be 1 for each (i,j,k), and this gives exactly 1. 
    # So rank 1 works?! That can't be right.
    
    # Oh wait, I think I'm confusing myself. Let me re-read the problem.
    # The tensor T[i,j,k] should equal 1 for all valid (i,j,k). The decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # If I set U[i*m+j,c] = 1 for all, V[j*p+k,c] = 1 for all, W[k*n+i,c] = 1 for all,
    # then T[i,j,k] = 1 for all (i,j,k). That's rank 1!
    
    # But that's clearly wrong because matrix multiplication has tensor rank > 1.
    # The issue: the tensor for matrix multiplication is NOT all ones. 
    # T[i,j,k] = 1 means C[i,k] = sum_j A[i,j]*B[j,k], so the tensor has T[i,j,k]=1 for all valid indices.
    # And indeed, the rank-1 decomposition above works!
    
    # Wait, but that would mean all matrix multiplications have rank 1, which is false.
    # I must be misunderstanding the tensor definition.
    
    # Let me reconsider: the multiplication tensor T has T[i,j,k] = 1 for all i,j,k in range.
    # The claim is that this tensor has rank 1 with the uniform vectors. Let me verify with a small case.
    # For 1x1x1: T[0,0,0] = 1. Rank 1: U[0,0]=1, V[0,0]=1, W[0,0]=1. Sum = 1. Correct.
    # For 2x1x1: T[0,0,0]=1, T[1,0,0]=1. 
    # Rank 1: U[i,0]=1, V[0,0]=1, W[0,i]=1. T[i,0,0] = 1*1*1 = 1. Correct!
    # For 1x2x1: T[0,0,0]=1, T[0,1,0]=1.
    # Rank 1: U[0,j]=1, V[j,0]=1, W[0,0]=1. T[0,j,0] = 1*1*1 = 1. Correct!
    # For 1x1x2: similar.
    # For 2x2x2: T[i,j,k]=1 for all. 
    # Rank 1: U[i,j]=1, V[j,k]=1, W[k,i]=1. T[i,j,k] = 1. Correct!
    
    # So the all-ones tensor of any size has rank 1?! That seems to contradict known results.
    # I think the issue is that the multiplication tensor is NOT the all-ones tensor.
    # Let me re-read the problem statement.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 6 matrix."
    # The tensor should encode the multiplication operation. The standard multiplication tensor T[i,j,k]
    # for C = A*B where C[i,k] = sum_j A[i,j]*B[j,k] has T[i,j,k] = 1 for all valid (i,j,k).
    # And I just showed this has rank 1. But that contradicts the fact that matrix multiplication
    # is believed to require more than rank 1.
    
    # I think I'm making an error. Let me look at this more carefully.
    # The tensor decomposition for bilinear maps: a bilinear map f: U x V -> W can be written as
    # f(u,v) = sum_c u^T U_c v V_c W_c^T or similar. The tensor T has T[i,j,k] such that
    # f(u,v)[k] = sum_{i,j} u[i] v[j] T[i,j,k].
    
    # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k].
    # Here u = A (flattened), v = B (flattened), and the output is C (flattened).
    # T[i,j,k] where i indexes A, j indexes B, k indexes C.
    # T[i,j,k] = 1 if the indices are consistent, i.e., if i is a row of A, j is a column of A / row of B,
    # and k is a column of B / row of C, AND the j in A matches the j in B, and the k in B matches the k in C.
    # In our case, all indices are "valid" by construction, so T[i,j,k] = 1 for all i,j,k.
    
    # But wait, the index j appears in both A and B. When we flatten A to a vector, index i*m+j corresponds
    # to A[i,j]. When we flatten B, index j*p+k corresponds to B[j,k]. The "j" must match.
    # So the tensor is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # And my rank-1 construction seems to work.
    
    # I must be fundamentally misunderstanding something. Let me check with a concrete example.
    # Take n=2, m=2, p=2. A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]].
    # C[0,0] = a11*b11 + a12*b21
    # C[0,1] = a11*b12 + a12*b22
    # C[1,0] = a21*b11 + a22*b21
    # C[1,1] = a21*b12 + a22*b22
    
    # The tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}.
    # My rank-1 claim: U[i,j] = 1, V[j,k] = 1, W[k,i] = 1.
    # Then sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1*1*1 = 1 for all (i,j,k). 
    # This says T[i,j,k] = 1, which is correct.
    
    # But this means the bilinear map is: C[k] = sum_{i,j} A[i,j] * B[j,k] * 1? No...
    # The reconstruction is: C_flat[k] = sum_{i,j} A_flat[i*m+j] * B_flat[j*p+k] * T[i,j,k]
    # = sum_{i,j} A[i,j] * B[j,k] * 1
    # = sum_j A[i,j] B[j,k] ... but this sums over i too!
    # C_flat[k] = sum_{i,j} A[i,j] B[j,k] = sum_j (sum_i A[i,j]) B[j,k]
    # That's NOT the same as matrix multiplication!
    
    # I see the error. The tensor T[i,j,k] should NOT be 1 for all (i,j,k). 
    # It should be 1 only when the indices are "consistent" in a specific way.
    # Actually, for matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) IS correct,
    # but the reconstruction formula is different.
    
    # The correct reconstruction: C[i,k] = sum_j A[i,j] B[j,k].
    # In tensor form: C[i,k] = sum_{i',j',k'} A[i',j'] B[j',k'] T[i',j',k'] * (delta_{i,i'} * delta_{k,k'})
    # No, that's not right either.
    
    # I think the issue is that the tensor T[i,j,k] for matrix multiplication is indeed 1 for all
    # valid indices, and the rank-1 decomposition I found is valid for THIS tensor. But this tensor
    # does NOT correctly represent matrix multiplication in the way I'm thinking.
    
    # Let me look at the problem statement again: "the output factor follows the official trace convention W[k*n+i,r]"
    # And the condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    # This IS the all-ones tensor, and it DOES have rank 1.
    
    # But that can't be the intended problem. Let me re-read: "Find a rank-minimal exact tensor decomposition
    # for multiplying a 2 by 3 matrix by a 3 by 6 matrix."
    
    # I think the tensor is NOT all ones. The tensor should encode the multiplication such that
    # the bilinear form gives the correct product. Let me think about what tensor that is.
    
    # The multiplication map is: (A, B) -> A*B. This is bilinear. The associated tensor T has
    # T[i,j,k] where i indexes the row of A and column of C, j indexes the column of A and row of B,
    # k indexes the column of B and column of C. For the bilinear map to compute C[i,k] = sum_j A[i,j]B[j,k],
    # we need T[i,j,k] = 1 for all valid (i,j,k). And the reconstruction is:
    # C[i,k] = sum_{j} A[i,j] B[j,k] T[i,j,k] = sum_j A[i,j] B[j,k].
    # So T IS the all-ones tensor, and it DOES have rank 1.
    
    # I'm clearly missing something fundamental. Let me just go with the schoolbook rank-36 solution
    # and accept that my understanding is flawed. The problem must define the tensor differently.
    
    # Given the confusion, let me just provide a correct schoolbook decomposition of rank 36.
    
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
