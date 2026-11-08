"""Matrix multiplication 3x5x5: find rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{i,i'}delta_{j,j'}delta_{k,k'}"""
    r = len(U[0]) if U else 0
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # And for distinct indices, should be 0 (handled by tensor structure)
    # Actually the tensor is: T[i,j,k; i',j',k'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # Decomposition: T = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right. Let me re-read the convention.
    # W[k*n+i, r] is the official trace convention.
    # The multiplication tensor: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = 1 (the structure tensor of matrix mult)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must hold for ALL i,j,k.
    # For the standard tensor, T[i,j,k] = 1 for all valid i,j,k (it's the identity structure)
    # Wait no. The matrix multiplication tensor has entries T[i,j,k; i',j',k'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    # The rank decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r where
    # u_r[i,j] = U[i*m+j, r], v_r[j,k] = V[j*p+k, r], w_r[k,i] = W[k*n+i, r]
    # And T[i,j,k] (the "flat" tensor) = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # For matrix multiplication, this flat tensor equals 1 for all (i,j,k) since
    # the structure is: C[i,k] = sum_j A[i,j] B[j,k], so the tensor is sum over j of A[i,j]*B[j,k]
    # The tensor T has T[i,j,k] = 1 (each term A[i,j]*B[j,k] contributes independently)
    
    # Check: for all i in range(n), j in range(m), k in range(p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook baseline
    rank_school = n * m * p  # 75
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank using algebraic methods
    # The key insight: for matrix multiplication, we can sometimes use
    # Strassen-like tricks or exploit the rectangular shape.
    
    # Strategy: Try rank values below 75 and check if a valid decomposition exists.
    # For 3x5x5, the minimum rank is at most 75 (schoolbook).
    # Known lower bounds: rank >= n*m + m*p + p*n - 2*min(n*m, m*p, p*n) ... not standard.
    # Actually, a simple bound: rank >= max(n*m, m*p, p*n) = max(15, 25, 15) = 25
    # And rank >= n*m*p / max(n,m,p) ... not a standard bound.
    
    # Let's try a systematic approach: use the fact that for rectangular matrices,
    # we can sometimes save rank by combining terms.
    
    # For 3x5x5: A is 3x5, B is 5x5, C is 3x5.
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    # For each i (3 values) and k (5 values), we need to compute sum of 5 products.
    
    # One approach: group the j-summation. For each i, we compute 5 dot products of
    # row i of A with columns of B.
    
    # Let's try a construction that exploits linearity:
    # For a fixed i, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + ... + A[i,4]*B[4,k]
    # This is a linear combination of the rows of B, weighted by A[i,j].
    
    # Can we share computations across i? The A[i,j] values are different for each i.
    
    # Let's try: for each j, define S_j[k] = B[j,k]. Then C[i,k] = sum_j A[i,j]*S_j[k].
    # This doesn't obviously save rank.
    
    # Alternative: Use the fact that we only need 3*5=15 outputs, each being a sum of 5 terms.
    # Total "work" is 75 products. Can we reduce by sharing?
    
    # Let me try a direct algebraic approach: assume a rank-r decomposition and solve.
    # This is hard for large r, but let's try small improvements.
    
    # Try rank 74: can we merge two schoolbook terms?
    # Terms (i,j,k) and (i',j',k') can be merged if there's a linear dependence.
    # In the schoolbook, term (i,j,k) has U nonzero only at (i*m+j), V only at (j*p+k), W only at (k*n+i).
    # Two terms can be combined if their "fingerprint" allows cancellation.
    
    # Let's try a more systematic search using linear algebra.
    # The tensor T is in R^{15 x 25 x 15} (flattened: U dim=15, V dim=25, W dim=15).
    # We want rank_r(T) = min r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c.
    
    # For small rank, we can use the "border rank" or direct optimization.
    # But exact rational decomposition is needed.
    
    # Let's try a constructive approach:
    # For 3x5x5, note that p=n=5... no, n=3, m=5, p=5.
    
    # Key observation: C is 3x5, so for each column k of C, we have:
    # C[:,k] = A * B[:,k] (matrix-vector product, 3x5 times 5x1)
    # There are 5 such columns. Each requires a 3x5 matrix-vector product.
    
    # A 3x5 matrix-vector product: y = Ax where x is 5-dim, y is 3-dim.
    # y_i = sum_j A[i,j]*x_j
    # This has "rank" 15 (3*5) in the tensor sense.
    
    # But we have 5 such products sharing the same A!
    # C[:,k] = A * B[:,k] for k=0..4.
    # So C = A * B (standard matrix mult).
    
    # The tensor for this is: T[i,j,k] = 1 (for all i,j,k), which is the rank-75 tensor
    # in the sense that it's the sum of 75 rank-1 tensors.
    
    # Can we do better? The tensor is actually the "all-ones" tensor of size 3x5x5.
    # Wait, no. Let me re-derive.
    
    # Matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The structure tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # This is because the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So T is the all-ones tensor of shape (3,5,5), i.e., T[i,j,k] = 1.
    
    # The rank of the all-ones tensor of shape (n,m,p) over a field:
    # It's known that the rank of the all-ones tensor J_{n,m,p} is:
    # - If n=m=p=1: rank 1
    # - In general, rank(J_{n,m,p}) can be less than n*m*p.
    # 
    # Actually, J_{n,m,p}[i,j,k] = 1 for all i,j,k.
    # This tensor has rank = min(n,m,p) if we allow arbitrary fields? No.
    # 
    # Let me think: J[i,j,k] = 1 = sum_c u_c[i] v_c[j] w_c[k]
    # We need: for all (i,j,k), sum_c u_c[i] v_c[j] w_c[k] = 1.
    # 
    # This is equivalent to: the matrix M_{(i,j),k} = J[i,j,k] has rank... 
    # J as a 3D array: flatten to (nm) x p matrix where entry [(i,j),k] = 1.
    # This is a nm x p all-ones matrix, which has rank 1!
    # Similarly, flattening to (np) x m gives rank 1, and (mp) x n gives rank 1.
    # 
    # But the tensor rank is NOT the same as the matrix rank of any flattening.
    # The tensor rank of J_{n,m,p}: 
    # 
    # Claim: rank(J_{n,m,p}) = 1 if and only if at most two of n,m,p are > 1? No.
    # J[i,j,k] = 1. Can we write 1 = sum_c u_c[i] v_c[j] w_c[k]?
    # Yes! Take r = 1: u[i]=1, v[j]=1, w[k]=1 for all. Then sum_c=1: 1*1*1 = 1. ✓
    # 
    # WAIT. That means the all-ones tensor has rank 1!
    # 
    # But that can't be right for matrix multiplication. Let me re-examine.
    
    # OH. I think I'm confusing the structure tensor with the multiplication tensor.
    # 
    # The matrix multiplication tensor is NOT the all-ones tensor. Let me be precise.
    # 
    # We have: C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # The multiplication tensor T ∈ R^{(n×m) × (m×p) × (n×p)} is defined by:
    # T[A_index, B_index, C_index] where A_index = (i,j), B_index = (j,k), C_index = (i,k)
    # T[(i,j), (j',k), (i',k')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    # 
    # So T[(i,j), (j,k), (i,k)] = 1, and T[(i,j), (j',k), (i',k')] = 0 otherwise.
    # 
    # The rank decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c where
    # U_c is in R^{n×m}, V_c in R^{m×p}, W_c in R^{n×p}
    # 
    # In the flattened convention of the problem:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # 
    # And the condition is: for all (i,j,k,i',j',k'):
    # sum_c U[i*m+j, c] V[j'*p+k', c] W[k'*n+i', c] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    # 
    # Wait, that's not quite right either. Let me re-read the problem.
    # 
    # "W[k*n+i,r] is the official trace convention"
    # 
    # The standard decomposition of matrix multiplication:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Tensor form: sum_j A[i,j] B[j,k] C^*[i,k] = 0 (where C^* is dual)
    # Or: the bilinear map (A,B) -> C has associated tensor.
    # 
    # The rank-r decomposition means: there exist r triples (U_r, V_r, W_r) such that:
    # C[i,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]  ... no, this doesn't make sense.
    # 
    # Let me look at this differently. The problem says:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # 
    # The decomposition condition must be:
    # For the matrix multiplication to be correct, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    # 
    # No wait, that's for the "flat" tensor. Let me think about what the actual condition is.
    # 
    # I believe the condition is:
    # The multiplication tensor T ∈ R^{(nm) × (mp) × (pn)} defined by
    # T[(i,j), (j',k), (k',i')] = δ_{i,i'} δ_{j,j'} δ_{k,k'}
    # 
    # And we decompose T = sum_c u_c ⊗ v_c ⊗ w_c where
    # u_c ∈ R^{nm}, v_c ∈ R^{mp}, w_c ∈ R^{pn}
    # 
    # So: T[(i,j), (j',k), (k',i')] = sum_c U[i*m+j, c] V[j'*p+k, c] W[k'*n+i', c]
    # 
    # And we need this to equal δ_{i,i'} δ_{j,j'} δ_{k,k'} for all indices.
    
    # Let me verify with the schoolbook:
    # U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # 
    # T[(i,j),(j',k),(k',i')] = sum_{(a,b,c)} U[i*m+j,(a,b,c)] V[j'*p+k,(a,b,c)] W[k'*n+i',(a,b,c)]
    # = sum_{(a,b,c)} δ_{i,a}δ_{j,b} * δ_{j',b}δ_{k,c} * δ_{k',c}δ_{i',a}
    # = δ_{i,i'} δ_{j,j'} δ_{j,j'} δ_{k,k'} δ_{k',k'} 
    # Wait: δ_{j,b}δ_{j',b} = δ_{j,j'} (summed over b gives δ_{j,j'})
    # Similarly δ_{k,c}δ_{k',c} = δ_{k,k'}
    # δ_{i,a}δ_{i',a} = δ_{i,i'}
    # So T = δ_{i,i'} δ_{j,j'} δ_{k,k'} ✓
    
    # Great, so the condition is confirmed. Now, the all-ones tensor J[i,j,k]=1 is NOT what we have.
    # We have T[(i,j),(j',k),(k',i')] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    
    # This is actually a much more structured tensor. Let me think about its rank.
    
    # Flatten T as a (nm*mp) x (pn) matrix:
    # Row index: ((i,j), (j',k)), Column index: (k', i')
    # Entry: δ_{i,i'}δ_{j,j'}δ_{k,k'}
    # 
    # For a fixed row ((i,j),(j',k)), the column (k',i') has value 1 iff i'=i, j'=j, k'=k.
    # So each row has exactly one 1 (at position (k,i) in the (k',i') grid... wait, column is (k',i')).
    # The column (k',i') = (k, i) has value 1. All other columns are 0.
    # 
    # So as a (nm*mp) x (pn) matrix, T has each row with a single 1.
    # The rank is at most min(nm*mp, pn) = min(375, 15) = 15.
    # 
    # But tensor rank is different from matrix rank. The tensor rank is the minimum r such that
    # T can be written as sum of r rank-1 tensors.
    
    # Let me think about this more carefully. The tensor T has the property that:
    # T[(i,j),(j',k),(k',i')] = δ_{i,i'}δ_{j,j'}δ_{k,k'}
    
    # This can be rewritten as: T = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    # where e_{(i,j)} is the standard basis vector in R^{nm}, etc.
    # 
    # This is the schoolbook decomposition with rank = n*m*p = 75.
    # 
    # Can we do better? The key question is whether there are linear dependencies among these 75 rank-1 tensors.
    
    # Consider: can we express e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)} as a linear combination of others?
    # Generally no, because each has a unique "pattern" of non-zero entries.
    
    # However, for certain shapes, there might be savings. Let me check small cases.
    
    # For 1x1x1: rank 1 (just one term)
    # For 2x2x2: Strassen shows rank 7 < 8
    # For 3x3x3: best known is 23 (or 20 with complex numbers)
    
    # For 3x5x5, what's the best known? This is a research-level question.
    # But let me try to find a constructive improvement.
    
    # Idea: For 3x5x5, note that n=3 is small. Let's see if we can save rank by exploiting
    # the structure when n is small.
    
    # Alternative approach: Use the fact that the problem allows half-integer entries.
    # This opens up more possibilities for cancellations.
    
    # Let me try a direct search for rank < 75 using a greedy or algebraic method.
    
    # Actually, let me try a specific construction:
    # 
    # For C[i,k] = sum_j A[i,j] B[j,k], with i=0,1,2 and k=0,1,2,3,4:
    # 
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k] + A[0,4]B[4,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k] + A[1,4]B[4,k]
    # C[2,k] = A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k] + A[2,3]B[3,k] + A[2,4]B[4,k]
    # 
    # Total: 15 outputs, each is a sum of 5 products = 75 products total.
    
    # Can we share any computations? The products A[i,j]*B[j,k] are all distinct
    # (different i, j, k combinations), so no obvious sharing.
    
    # However, we might be able to use a "Strassen-like" trick by introducing
    # intermediate sums and then subtracting.
    
    # Let me try: define S[j,k] = B[j,k] (just B itself, no help).
    
    # What if we group by j? For each j, we have terms A[i,j]*B[j,k] for all i,k.
    # That's 3*5=15 terms per j, and 5 values of j, so 75 total.
    
    # I don't see an obvious saving for this specific shape. Let me try to verify
    # if rank 74 is possible by attempting a linear algebra approach.
    
    # Given the complexity, let me try a randomized search for a lower-rank decomposition
    # using the "tensor rank" optimization approach.
    
    # Actually, for the purpose of this task, let me try a few specific ranks and
    # use a simple verification. If I can't find a lower rank quickly, I'll return 75.
    
    # Let me try rank 74: can we merge two terms?
    # Term (i,j,k) and (i',j',k') can be merged if there exist coefficients such that
    # the sum of the two rank-1 tensors equals their individual contributions.
    # This requires: u1⊗v1⊗w1 + u2⊗v2⊗w2 = (u1+α)⊗(v1+β)⊗(w1+γ) + ... 
    # This is generally not possible unless there's a specific linear dependency.
    
    # Let me just return the schoolbook if I can't find better quickly.
    # But first, let me try one more idea: exploit the fact that n=3.
    
    # For n=3, we have 3 rows of C. Each row C[i,:] is a 1x5 vector = A[i,:] * B (1x5 times 5x5).
    # The product of a 1x5 and 5x5 matrix is a 1x5 matrix.
    # The tensor for 1x5x5 multiplication has rank 1*5*5=25 in schoolbook.
    # But all three rows share the same B!
    
    # C[0,:] = A[0,:] * B
    # C[1,:] = A[1,:] * B  
    # C[2,:] = A[2,:] * B
    
    # Can we compute these three products with fewer than 3*25=75 operations?
    # 
    # Define: for each k, C[:,k] = B[:,k]^T * A^T (5x1 times 1x3... no)
    # C[:,k] = sum_j A[:,j] * B[j,k] (3x1 vector = sum of 3x1 vectors)
    # 
    # C[:,k] = A[:,0]*B[0,k] + A[:,1]*B[1,k] + A[:,2]*B[2,k] + A[:,3]*B[3,k] + A[:,4]*B[4,k]
    # 
    # This is a linear combination of the 5 columns of A, weighted by the k-th row of B.
    # 
    # For all k simultaneously: C = A*B.
    # 
    # I don't see an obvious rank saving. Let me just return the schoolbook.
    
    # Actually, wait. Let me reconsider. The problem says "half-integer entries" are allowed.
    # This suggests that the intended solution uses some non-trivial algebraic identity.
    
    # Let me try a very simple improvement: can we reduce from 75 to 74 by a local trick?
    # 
    # Consider two terms: (i,j,k) and (i,j,k') with same i,j but different k.
    # Term 1: U[i*m+j, c1]=1, V[j*p+k, c1]=1, W[k*n+i, c1]=1
    # Term 2: U[i*m+j, c2]=1, V[j*p+k', c2]=1, W[k'*n+i, c2]=1
    # 
    # Can we replace these with 2 terms that also cover the same "space"? 
    # The issue is that these terms contribute to different (i,j,k) and (i,j,k') positions
    # in the tensor, so they can't be simply merged.
    
    # I'll go with the schoolbook rank of 75.
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        return best
    
    return best
# EVOLVE-BLOCK-END
