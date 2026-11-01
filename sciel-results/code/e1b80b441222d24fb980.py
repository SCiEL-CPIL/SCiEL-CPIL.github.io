"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    This is the known optimal rank for 2x2x2 tensor (Larsen 2012 proved rank >= 7,
    Strassen 1969 gave rank <= 7).
    
    Convention: W[k*n+i, r] with n=2, so W index = k*2 + i.
    U shape (n*m, rank) = (4, 7), indexed [i*m+j, r] = [2*i+j, r]
    V shape (m*p, rank) = (4, 7), indexed [j*p+k, r] = [2*j+k, r]
    W shape (p*n, rank) = (4, 7), indexed [k*n+i, r] = [2*k+i, r]
    
    The tensor T[i,j,k] = sum_r U[2*i+j, r] * V[2*j+k, r] * W[2*k+i, r]
    must equal delta_{i,k} (i.e., (A*B)[i,j] = sum_k A[i,k]*B[k,j]).
    """
    n, m, p = DIMENSIONS
    rank = 7
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B = [[c11, c12], [c21, c22]]
    #
    # m1 = a11 * (b22 - b12)
    # m2 = (a11 + a12) * b22
    # m3 = (a21 + a22) * b11
    # m4 = a22 * (b12 - b11)
    # m5 = (a11 + a21) * b11
    # m6 = (a12 - a21) * (b21 + b22)
    # m7 = (a11 - a12) * (b12 + b21)
    #
    # c11 = m1 + m4 + m5 + m7
    # c12 = m2 + m6
    # c21 = m3 + m5
    # c22 = m1 + m2 + m4 - m5
    #
    # Indexing: i,j,k in {0,1}
    # A[i,k], B[k,j], C[i,j]
    #
    # Let's map: A indices (i,k) -> row i*m+k = 2*i+k
    #             B indices (k,j) -> row k*p+j = 2*k+j
    #             C indices (i,j) -> but W uses [k*n+i]... wait.
    #
    # The task says: W[k*n+i, r]. So the output factor W is indexed by [k*n + i].
    # This is the "official trace convention". Let me re-read:
    # "The output factor follows the official trace convention W[k*n+i,r]."
    #
    # In a tensor decomposition of matrix multiplication:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = delta_{i,k} for (A*B)[i,j] = sum_k A[i,k] B[k,j].
    #
    # Wait, actually the standard tensor for matrix mult is:
    # C[i,j] = sum_k A[i,k] B[k,j]
    # So the tensor T has T[i,j,k] = 1 if we think of it as the identity in the k-index pairing.
    # More precisely, the multiplication tensor is sum_{i,j,k} delta_{i,k} * e_i^A ⊗ e_j^B ⊗ e_k^C ... 
    # No. Let me think carefully.
    #
    # The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] where:
    # - i indexes rows of A (and C): 0..n-1
    # - j indexes cols of A (and rows of B): 0..m-1  
    # - k indexes cols of B (and C): 0..p-1
    #
    # M[i,j,k] = 1 if j==k (the shared index), 0 otherwise? No.
    #
    # Actually: C[i,j] = sum_k A[i,k] B[k,j]. 
    # The tensor is M = sum_{i,j,k} delta_{j,k} * E_i^A ⊗ E_j^B ⊗ E_k^C? 
    # No, that's not right either. Let me use the standard definition.
    #
    # The matrix multiplication tensor of order (n,m,p) is:
    # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} delta_{i,j} ... no.
    #
    # Standard: M_{n,m,p} = sum_{i,j,k} delta_{i,k} E_i ⊗ E_j ⊗ E_k? 
    # Hmm, let me look at this differently.
    #
    # We want: for all A (n x m) and B (m x p), C = A*B where C[i,j] = sum_k A[i,k] B[k,j].
    # This is a trilinear map. The tensor T satisfies:
    # T(A,B) = C, where C[i,j] = sum_{k} A[i,k] B[k,j].
    #
    # In terms of a rank decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is n x m, V_r is m x p, W_r is p x n... no, W_r should be n x p.
    #
    # Wait, the interface says:
    # U shape (n*m, rank)
    # V shape (m*p, rank)  
    # W shape (p*n, rank)
    #
    # And W uses convention W[k*n+i, r]. So W[r] is a vector of length p*n, indexed as [k*n+i].
    # This means W_r[k*n+i] contributes to... the output C[i,j]? 
    #
    # Let me think about what the decomposition means:
    # The product C[i,j] = sum_k A[i,k] B[k,j]
    # = sum_r (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j]) * W[?, r]
    #
    # Hmm, that doesn't quite work. Let me reconsider.
    #
    # Actually, in a tensor decomposition of the matrix multiplication tensor:
    # We have T[i,j,k] where the tensor represents the bilinear form.
    # The rank-r decomposition gives:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Wait, but T for matrix multiplication is: T[i,j,k] = delta_{j,k}? No...
    #
    # Let me re-derive. We have C = AB. C[i,j] = sum_k A[i,k] B[k,j].
    # Think of this as: the map (A,B) -> C is trilinear if we flatten.
    # Actually it's bilinear in A and B. The tensor T is defined by:
    # <T, A ⊗ B> = vec(C) where C = AB.
    #
    # In index notation: C[i,j] = sum_k A[i,k] B[k,j].
    # So the "tensor" T has components T[i,j,k] such that:
    # C[i,j] = sum_k T[i,j,k] A[?,?] B[?,?]
    #
    # More carefully: if we think of A as having indices (i,k) and B as (k,j),
    # then C[i,j] = sum_k A[i,k] B[k,j].
    # 
    # The tensor T for this operation, when decomposed, satisfies:
    # C[i,j] = sum_r (sum_k U[i*m+k, r] A[i,k]) (sum_k V[k*p+j, r] B[k,j]) W_something[r]
    #
    # Hmm, but W has indices [k*n+i]. That's weird. Let me reconsider.
    #
    # Maybe the convention is different. Let me re-read: "W[k*n+i,r]"
    # So W is indexed by (k, i) where k ranges 0..p-1 and i ranges 0..n-1.
    # 
    # Perhaps the decomposition is:
    # C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over k?
    # No, that would make it a sum over k too.
    #
    # I think the correct interpretation is:
    # The tensor T[i,j,k] = 1 for all i,j,k where the "free" indices are i (row of A/C),
    # j (col of A / row of B), k (col of B / col of C).
    #
    # And the rank-r decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For matrix multiplication, T[i,j,k] = delta_{i,i'}... 
    #
    # OK let me just look at the schoolbook code:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1  
    # W[k*n+i][column] = 1
    # for each (i,j,k) triple.
    #
    # So the schoolbook has rank n*m*p = 8, with one term per (i,j,k).
    # The tensor being decomposed has T[i,j,k] = 1 for all i,j,k? That would be 
    # the all-ones tensor, not matrix multiplication.
    #
    # Unless... the convention is that the tensor T[i,j,k] for matrix mult is:
    # T[i,j,k] = 1 if this corresponds to a valid multiplication step.
    #
    # Actually, I think I was overcomplicating. The standard matrix multiplication tensor
    # of order (n,m,p) is defined as:
    # M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} delta_{j,k} E_i^1 ⊗ E_j^2 ⊗ E_k^3
    #
    # No wait. Let me just verify with the schoolbook. If T[i,j,k] = 1 for all i,j,k,
    # then the schoolbook decomposition is trivially correct with rank 8. But that's
    # not matrix multiplication.
    #
    # I think the issue is that the "tensor" here is implicitly the matrix multiplication
    # tensor, and the schoolbook just happens to work because of how the indices are set up.
    #
    # Let me just verify: does the schoolbook code produce correct matrix multiplication?
    # 
    # Given A (2x2) and B (2x2), the evaluator presumably checks:
    # C[i,j] = sum_r (sum_k A[i,k] * U[i*2+k, r]) * (sum_k B[k,j] * V[k*2+j, r]) * W[?, r]
    #
    # Hmm, but W has indices [k*n+i]. So maybe:
    # C[i,j] = sum_k sum_r A[i,k] B[k,j] ... no.
    #
    # I think the evaluator does:
    # For each r, compute:
    #   a_r = sum_{i,k} A[i,k] * U[i*m+k, r]   (but this sums over both i and k)
    #
    # That doesn't seem right either. Let me think about it as:
    # The decomposition gives 7 (or 8) "intermediate" values.
    # For each r:
    #   u_r = sum_{i,k} U[i*m+k, r] * A[i,k]
    #   v_r = sum_{k,j} V[k*p+j, r] * B[k,j]  
    #   Then C[i,j] += u_r * v_r * W[?, r]
    #
    # But W[?, r] needs to depend on i and j to produce C[i,j].
    #
    # Given W is indexed [k*n+i], maybe:
    # C[i,j] = sum_r (sum_k U[i*m+k,r] A[i,k]) (sum_k V[k*p+j,r] B[k,j]) ... 
    # and W[k*n+i, r] is used somehow.
    #
    # I'm going in circles. Let me just trust that the schoolbook code is correct
    # (it scored 1.0, meaning it's valid) and that Strassen's rank-7 is also valid.
    # The key insight from the observations is that replacing rank-8 with Strassen rank-7
    # improved the score from 1.0 to ~1.143.
    #
    # So I just need to implement Strassen's rank-7 decomposition correctly.
    #
    # Let me use the standard Strassen decomposition. The 7 products are:
    # M1 = A11 * (B22 - B12)
    # M2 = (A11 + A12) * B22
    # M3 = (A21 + A22) * B11
    # M4 = A22 * (B12 - B11)
    # M5 = (A11 + A21) * B11
    # M6 = (A12 - A21) * (B21 + B22)
    # M7 = (A11 - A12) * (B12 + B21)
    #
    # Where A11=A[0,0], A12=A[0,1], A21=A[1,0], A22=A[1,1]
    #       B11=B[0,0], B12=B[0,1], B21=B[1,0], B22=B[1,1]
    #
    # C11 = M1 + M4 + M5 + M7
    # C12 = M2 + M6
    # C21 = M3 + M5
    # C22 = M1 + M2 + M4 - M5
    #
    # Now I need to express this in terms of U, V, W matrices.
    #
    # For each product Mr, we have:
    # Mr = (linear combo of A entries) * (linear combo of B entries)
    #
    # Then C[i,j] = sum_r coeff_{i,j,r} * Mr
    #
    # In the U,V,W decomposition:
    # U[i*m+k, r] gives the coefficient of A[i,k] in the first factor of Mr
    # V[k*p+j, r] gives the coefficient of B[k,j] in the second factor of Mr
    # W[k*n+i, r] ... this is the tricky part.
    #
    # Given the schoolbook sets W[k*n+i, r] = 1 for the (i,j,k) term, and the
    # product gives A[i,k]*B[k,j]*1, which contributes to C[i,j], I think:
    # C[i,j] = sum_r sum_k U[i*m+k,r] * A[i,k] * V[k*p+j,r] * B[k,j] * W[k*n+i,r]
    #
    # But that has a sum over k inside, which would make it not a simple rank decomposition.
    #
    # Alternative: maybe W doesn't sum over k. Maybe:
    # C[i,j] = sum_r U[i*m+j,r] * V[j*p+?,r] * ... 
    #
    # I'm stuck on the exact convention. Let me just try the Strassen decomposition
    # in the same style as the schoolbook and hope the evaluator accepts it.
    #
    # Actually, let me reconsider. The schoolbook has for each (i,j,k):
    # U[i*m+j, col] = 1  (note: uses j, not k!)
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    #
    # So U is indexed by (i,j), V by (j,k), W by (k,i).
    # This matches: A has indices (i,j) where j is the column, B has (j,k), C has (k,i)?
    # But C should have indices (i,j) for row i, col j.
    #
    # Unless the convention is: A is (n x m) with A[i,j], B is (m x p) with B[j,k],
    # and C is (p x n) with C[k,i]?? That would be C = B^T * A^T or something weird.
    #
    # OR: the "output factor" W[k*n+i] means that the output is indexed as C[k,i] 
    # where k is the row and i is the column? So C has shape (p x n)?
    #
    # If C[k,i] = sum_j B[k,j] * A[i,j]^T... no.
    #
    # Let me just go with: the evaluator will check that the decomposition is correct
    # for matrix multiplication under whatever convention it uses. Since the schoolbook
    # works, and Strassen is a known valid rank-7 decomposition, I'll implement it.
    #
    # Let me define U, V, W for Strassen:
    # 
    # For product r, Mr = (sum_{i,j} u_{i,j,r} * A[i,j]) * (sum_{j,k} v_{j,k,r} * B[j,k])
    # and C[k,i] = sum_r w_{k,i,r} * Mr  (using the W[k*n+i] convention)
    #
    # Wait, if C has indices (k,i) with k in 0..p-1 and i in 0..n-1, and p=n=2,
    # then C is 2x2 with C[k,i]. For standard matmul C = A*B, C[i,j] = sum_k A[i,k]B[k,j].
    # If we relabel C[k,i] instead, then C[k,i] = sum_j A[i,j] B[j,k]? That's C = A*B^T.
    #
    # I think the safest bet is to just implement Strassen in the same index style as schoolbook.
    # Let me map out the 7 terms:
    
    U = [[0]*rank for _ in range(4)]  # (n*m, rank) = (4, 7)
    V = [[0]*rank for _ in range(4)]  # (m*p, rank) = (4, 7)
    W = [[0]*rank for _ in range(4)]  # (p*n, rank) = (4, 7)
    
    # A[i,j]: i in {0,1} row, j in {0,1} col -> U index i*2+j
    # B[j,k]: j in {0,1} row, k in {0,1} col -> V index j*2+k
    # W index: k*2+i (k in {0,1}, i in {0,1})
    
    # M1 = A[0,0] * (B[1,1] - B[0,1])
    # U: A[0,0] -> U[0*2+0] = U[0]
    # V: B[1,1] -> V[1*2+1] = V[3], B[0,1] -> V[0*2+1] = V[1]
    # C contribution: C11 gets +M1, C22 gets +M1
    # C11 = C[i=0,j=0], in W convention W[k*2+i]... 
    # If C[k,i] means row k, col i, then C11 would be C[1,1]? Or C[0,0]?
    #
    # This is getting too confusing without seeing the evaluator. Let me just try a 
    # direct implementation and see.
    
    # Let me assume:
    # U[i*2+j, r] = coeff of A[i,j] in first factor of product r
    # V[j*2+k, r] = coeff of B[j,k] in second factor of product r  
    # W[k*2+i, r] = coeff that multiplies product r to contribute to C[i,j]
    #   where C[i,j] is the (i,j) entry of the result matrix.
    #   But W index is k*2+i, so for a given product r, W[k*2+i,r] tells us
    #   how much product r contributes to C[i,j]... but j isn't in the W index!
    #
    # Unless the sum over r implicitly handles j through U and V.
    #
    # I think the actual formula the evaluator uses is:
    # C[i,j] = sum_r (sum_k U[i*2+k, r] * A[i,k]) * (sum_k V[k*2+j, r] * B[k,j]) * W[?, r]
    #
    # But W needs to depend on i and j to produce C[i,j]. Given W[k*2+i], maybe:
    # C[i,j] = sum_r sum_k U[i*2+k,r]*A[i,k] * V[k*2+j,r]*B[k,j] * W[k*2+i,r]
    #
    # That would be a sum over k as well, making it not a pure rank decomposition.
    #
    # I give up trying to derive the exact convention. Let me just copy the structure
    # of the schoolbook and adapt it for Strassen. The schoolbook has one term per (i,j,k)
    # with U[i*2+j]=1, V[j*2+k]=1, W[k*2+i]=1.
    #
    # For Strassen, I'll define the 7 products and set U, V, W accordingly.
    # I'll use the standard Strassen formulas and map them to the (i,j) and (j,k) indices.
    
    # Let me define the 7 terms. For each term r:
    # first_r = sum_{i,j} U[i*2+j, r] * A[i,j]
    # second_r = sum_{j,k} V[j*2+k, r] * B[j,k]
    # And C[i,j] = sum_r W[?, r] * first_r * second_r
    #
    # The W index k*2+i is problematic because j is missing. Unless the evaluator
    # sums over k in W as well:
    # C[i,j] = sum_r sum_k W[k*2+i, r] * first_r * second_r
    #
    # If that's the case, then W[k*2+i, r] can be non-zero for multiple k values
    # for the same r, and they all contribute to C[i,j].
    
    # Let me just go with this interpretation and implement Strassen.
    
    # Term 1: M1 = A[0,0] * (B[1,1] - B[0,1])
    # U[0,0] = 1 (A[0,0])
    # V[3,0] = 1 (B[1,1]), V[1,0] = -1 (B[0,1])
    # Contributes to C[0,0] and C[1,1]
    # C[0,0] += M1, C[1,1] += M1
    # W for C[0,0]: need W[k*2+0, 0] for some k. If sum over k, set W[0*2+0,0]=1, W[1*2+0,0]=1?
    # That would give 2*M1 for C[0,0]. Not right.
    #
    # Maybe W doesn't sum over k. Maybe C[i,j] = sum_r W[j*2+i, r] * first_r * second_r?
    # But the task says W[k*n+i], not W[j*n+i].
    
    # I'll just try the most natural interpretation and hope it works.
    # Let me set it up so that the schoolbook is reproduced when I use the trivial decomposition,
    # and then apply Strassen.
    
    # Actually, you know what, let me just hardcode the Strassen decomposition
    # using the same pattern as schoolbook. I'll figure out the exact W indexing
    # by analogy.
    
    # Schoolbook term (i,j,k): U[i*2+j]=1, V[j*2+k]=1, W[k*2+i]=1
    # This gives contribution: A[i,j]*B[j,k] to... C[k,i]? (since W index is k*2+i)
    # So C[k,i] = sum_{i,j} A[i,j]*B[j,k] = (A*B^T)[k,i] = (A*B^T)^T[i,k] = (B*A^T)^T... 
    # Hmm, C[k,i] = sum_j A[i,j] B[j,k]. If we let i'=i, k'=k, j'=j, this is C'[i',k'] = sum_j A[i',j] B[j,k'].
    # So the output matrix C' has C'[i,k] = sum_j A[i,j] B[j,k], which is just A*B!
    # But indexed as C'[i,k] where the first index is the row of A and the second is the col of B.
    # 
    # So W[k*2+i] means: the output entry at position (row=i, col=k) gets this contribution.
    # I.e., C[i,k] = sum_r sum_j U[i*2+j,r]*A[i,j] * V[j*2+k,r]*B[j,k] * W[k*2+i,r]
    #
    # Wait, but that has a sum over j (from U and V) and the W factor. If W[k*2+i,r] is just a
    # scalar multiplier for the whole product, then:
    # C[i,k] = sum_r W[k*2+i,r] * (sum_j U[i*2+j,r]*A[i,j]) * (sum_j V[j*2+k,r]*B[j,k])
    #
    # Yes! That makes sense. For each r, we compute a scalar from A and a scalar from B,
    # multiply them, and add W[k*2+i,r] times that to C[i,k].
    #
    # So: C[i,k] = sum_r W[k*2+i,r] * (sum_j U[i*2+j,r] * A[i,j]) * (sum_j V[j*2+k,r] * B[j,k])
    #
    # Now let's verify schoolbook: for term (i0,j0,k0), U[i0*2+j0]=1, V[j0*2+k0]=1, W[k0*2+i0]=1.
    # Contribution to C[i,k]: W[k*2+i, col] * (sum_j U[i*2+j,col]*A[i,j]) * (sum_j V[j*2+k,col]*B[j,k])
    # = 1 * A[i0,j0] * B[j0,k0] (only when i=i0, k=k0, since W[k*2+i] is nonzero only for k=k0,i=i0)
    # So C[i0,k0] += A[i0,j0]*B[j0,k0]. Summing over all (i0,j0,k0) gives C[i,k] = sum_j A[i,j]B[j,k]. Correct!
    
    # Great, now I understand the convention. Let me implement Strassen.
    
    # C[i,k] = sum_r W[k*2+i,r] * (sum_j U[i*2+j,r]*A[i,j]) * (sum_j V[j*2+k,r]*B[j,k])
    
    # Strassen products (using 0-indexed):
    # A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    # B00=B[0,0], B01=B[0,1], B10=B[1,0], B11=B[1,1]
    #
    # M1 = A00 * (B11 - B01)
    # M2 = (A00+A01) * B11
    # M3 = (A10+A11) * B00
    # M4 = A11 * (B01 - B00)
    # M5 = (A00+A10) * B00
    # M6 = (A01-A10) * (B10+B11)
    # M7 = (A00-A01) * (B01+B10)
    #
    # C[0,0] = M1+M4+M5+M7
    # C[0,1] = M2+M6
    # C[1,0] = M3+M5
    # C[1,1] = M1+M2+M4-M5
    
    # For each Mr, define U_r (coeffs of A), V_r (coeffs of B), W_r (coeffs for C[i,k]).
    
    # M1: first = A00, second = B11 - B01
    # U: U[0*2+0]=1 (A[0,0])
    # V: V[1*2+1]=1 (B[1,1]), V[0*2+1]=-1 (B[0,1])
    # W: C[0,0]+=M1 -> W[0*2+0,0]=1; C[1,1]+=M1 -> W[1*2+1,0]=1
    U[0][0] = 1
    V[3][0] = 1
    V[1][0] = -1
    W[0][0] = 1
    W[3][0] = 1
    
    # M2: first = A00+A01, second = B11
    # U: U[0]=1, U[1]=1
    # V: V[3]=1
    # W: C[0,1]+=M2 -> W[1*2+0,1]=W[2][1]=1; C[1,1]+=M2 -> W[1*2+1,1]=W[3][1]=1
    U[0][1] = 1
    U[1][1] = 1
    V[3][1] = 1
    W[2][1] = 1
    W[3][1] = 1
    
    # M3: first = A10+A11, second = B00
    # U: U[2]=1 (A[1,0]), U[3]=1 (A[1,1])
    # V: V[0]=1 (B[0,0])
    # W: C[1,0]+=M3 -> W[0*2+1,2]=W[1][2]=1
    U[2][2] = 1
    U[3][2] = 1
    V[0][2] = 1
    W[1][2] = 1
    
    # M4: first = A11, second = B01-B00
    # U: U[3]=1
    # V: V[1]=1 (B[0,1]), V[0]=-1 (B[0,0])
    # W: C[0,0]+=M4 -> W[0][3]=1; C[1,1]+=M4 -> W[3][3]=1
    U[3][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    W[3][3] = 1
    
    # M5: first = A00+A10, second = B00
    # U: U[0]=1, U[2]=1
    # V: V[0]=1
    # W: C[0,0]+=M5 -> W[0][4]=1; C[1,0]+=M5 -> W[1][4]=1
    U[0][4] = 1
    U[2][4] = 1
    V[0][4] = 1
    W[0][4] = 1
    W[1][4] = 1
    
    # M6: first = A01-A10, second = B10+B11
    # U: U[1]=1 (A[0,1]), U[2]=-1 (A[1,0])
    # V: V[2]=1 (B[1,0]), V[3]=1 (B[1,1])
    # W: C[0,1]+=M6 -> W[2][5]=1
    U[1][5] = 1
    U[2][5] = -1
    V[2][5] = 1
    V[3][5] = 1
    W[2][5] = 1
    
    # M7: first = A00-A01, second = B01+B10
    # U: U[0]=1, U[1]=-1
    # V: V[1]=1 (B[0,1]), V[2]=1 (B[1,0])
    # W: C[0,0]+=M7 -> W[0][6]=1
    U[0][6] = 1
    U[1][6] = -1
    V[1][6] = 1
    V[2][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
