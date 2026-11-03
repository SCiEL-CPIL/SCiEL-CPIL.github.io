"""Matrix multiplication 2x3x5 rank minimization."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n  # 6, 15, 10
    
    # Build the tensor T as a 3-way array T[a,b,c] for a in [N], b in [M], c in [P]
    # where T[i,j,k] = 1 if (i,j,k) corresponds to a valid (i_row, j, k_col) in the multiplication.
    # Actually, for matrix multiplication: C[i][k] = sum_j A[i][j] * B[j][k]
    # The tensor T is a 3-index array where T[i, j, k] = 1 if the product term (A[i,j]*B[j,k]) contributes to C[i,k]
    # So T[i, j, k] = 1 for all valid i, j, k combinations.
    # The rank is the minimum r such that T = sum_c u_c(a) * v_c(b) * w_c(c)
    
    # T has dimensions N x M x P = 6 x 15 x 10
    # T[i,j,k] = 1 if i < n, j < m, k < p (they all are)
    # Actually every triple (i, j, k) where i in [0,n), j in [0,m), k in [0,p) gives T[i,j,k] = 1
    # Because for each i, j, k the term A[i,j]*B[j,k] contributes to C[i,k].
    
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is a rank 3 tensor that's the outer product of three vectors all of ones? No.
    # T = sum over all (i,j,k) of E_i * E_j * E_k? No, that would be rank N*M*P.
    # Actually T[i,j,k] = 1 means T = 1_{n} ⊗ 1_{m} ⊗ 1_{p} where these are all-ones vectors.
    # Wait, that's rank 1! But that can't be right because matrix multiplication requires summing over j.
    
    # Let me reconsider. The tensor for matrix multiplication is:
    # T_{i,j,k} = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # The "term" is indexed by (i, j, k) and contributes 1 * A[i,j] * B[j,k] to C[i,k]
    # So the decomposition is: T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # For schoolbook: each (i,j,k) triple gets its own coefficient, so U[i,c] = 1, V[j,c] = 1, W[k,c] = 1 for the corresponding c.
    
    # So indeed T[i,j,k] = 1 for all valid triples, and we want to write this as a sum of rank-1 tensors.
    # But T is NOT rank 1 because... wait, is it?
    # T[i,j,k] = 1 for all i, j, k. This IS the all-ones tensor, which has rank 1!
    # U[i,c] = 1, V[j,c] = 1, W[k,c] = 1 for c=0, everything else 0.
    # Then sum_c U[i,c]*V[j,c]*W[k,c] = 1 for all i,j,k. Yes!
    
    # But wait, that's not right either. In matrix multiplication, we need:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition gives: (A*B)[i,k] = sum_c (sum_j U[i,c] * A[i,j]... no.
    
    # Let me re-read the task. The decomposition is:
    # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    # And the multiplication is: C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * (something)
    
    # Actually, I think the standard formulation is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_c (sum_i' U[i',c] A[i',j]... ) -- no.
    
    # Let me think again. In Strassen's formulation:
    # We have matrices A (n x m) and B (m x p).
    # We compute intermediate products: for c in range(rank):
    #   X_c = (U_c^T A) - this doesn't work dimensionally.
    
    # The correct formulation: T is a 3D tensor of size n x m x p where T[i,j,k] = 1.
    # The rank-r decomposition gives: T[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
    # Then: C[i,k] = sum_c (sum_j A[i,j] V[j,c]) * W[k,c] * ... 
    
    # Actually I think the convention is:
    # (A·B)[i,k] = Σ_{j,c} A[i,j] · (V[j,c]) · ... no.
    
    # Let me just use the schoolbook answer. The task says W follows convention W[k*n+i, r].
    # In the seed code, W is indexed as W[k*n + i][c], so W has shape (p*n, rank).
    # U has shape (n*m, rank), V has shape (m*p, rank).
    
    # The check should be: for all i,j,k:
    # sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = 1
    
    # With schoolbook: each (i,j,k) gets a unique c, and U[i*m+j,c] = V[j*p+k,c] = W[k*n+i,c] = 1.
    # So the sum is 1 for each (i,j,k). Correct.
    
    # Now, can we do better than rank 30?
    # T[i,j,k] = 1 for all (i,j,k) in {0..n-1} x {0..m-1} x {0..p-1}
    # This is the all-ones tensor of size n×m×p, which has tensor rank 1!
    # Because T[i,j,k] = u[i] * v[j] * w[k] where u[i]=1, v[j]=1, w[k]=1.
    
    # Wait, but that can't be right for matrix multiplication. Let me re-examine.
    
    # Ah, I see the issue. The tensor isn't T[i,j,k]=1 for all i,j,k.
    # The tensor encodes HOW the scalar products contribute to the output.
    
    # For matrix multiplication C = A·B where A is n×m and B is m×p:
    # C[i,k] = Σ_j A[i,j] B[j,k]
    # 
    # In the tensor decomposition framework, we want:
    # C[i,k] = Σ_c Σ_j (Σ_i' ... ) 
    # 
    # Actually, the standard way: define the tensor τ with τ[i,j,k] = 1 for the specific 
    # relationship that (i,j) from A and (j,k) from B produce output (i,k).
    # Since EVERY (i,j,k) triple where 0≤i<n, 0≤j<m, 0≤k<p is a valid contribution,
    # τ[i,j,k] = 1 for all such triples.
    # 
    # And τ IS the all-ones tensor, which has rank 1.
    
    # But that would mean rank 1 matrix multiplication, which is absurd.
    
    # I think I'm confusing myself. Let me look at this differently.
    
    # The issue is that in the rank-1 case with U[i,c]=V[j,c]=W[k,c]=1:
    # The computed value would be (Σ_j A[i,j]) * 1 * (Σ_k' ...) -- no.
    
    # Let me trace through the formula properly.
    # If T[i,j,k] = Σ_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # and for rank 1 with U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1:
    # T[i,j,k] = 1 for all i,j,k.
    # 
    # The multiplication algorithm using this decomposition:
    # For each c:
    #   P_c = 0
    #   For i in range(n): For j in range(m):
    #     P_c += A[i,j] * U[i*m+j, c]  -- but this isn't quite right either.
    
    # I think the correct interpretation is:
    # C[i,k] = Σ_c (Σ_i' Σ_j' U[i'... 
    # 
    # OK let me just look at it from the Strassen algorithm perspective.
    # Strassen for 2x2: C = A·B where A, B are 2x2.
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 − B22)  
    # M4 = A22(B21 − B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 − A11)(B11 + B12)
    # M7 = (A12 − A22)(B21 + B22)
    # C11 = M1 + M4 − M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 − M2 + M3 + M6
    # 
    # Each M_c is a scalar product of linear combinations of A entries and B entries.
    # U matrix: for each c, U[i*m+j, c] is the coefficient of A[i,j] in the "A-part" of M_c
    # V matrix: for each c, V[j*p+k, c] is the coefficient of B[j,k] in the "B-part" of M_c
    # W matrix: for each c, W[k*n+i, c] is the coefficient of M_c in computing C[i,k]
    
    # And the decomposition constraint is:
    # For the multiplication to work: Σ_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i, j, k (because A[i,j]*B[j,k] must appear with coefficient 1 in C[i,k]).
    
    # So yes, the constraint IS: Σ_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k).
    
    # And the all-ones tensor satisfies this with rank 1! But that's wrong because
    # it would mean C[i,k] = (Σ_j A[i,j]) * (Σ_k B[?,k])... no.
    
    # Wait, if U[i*m+j, 0] = 1 for all i,j, V[j*p+k, 0] = 1 for all j,k, W[k*n+i, 0] = 1 for all k,i:
    # Then C[i,k] computed via the decomposition = Σ_c W[k*n+i, c] * P_c where P_c = Σ_j' Σ_k' ... 
    # No, P_c is computed as: P_c = (some linear comb of A) * (some linear comb of B)
    # Specifically: P_c = (Σ_{i,j} U[i*m+j, c] * A[i,j]) * (Σ_{j,k} V[j*p+k, c] * B[j,k])
    # Then C[i,k] = Σ_c W[k*n+i, c] * P_c
    
    # With all-ones: P_0 = (Σ A) * (Σ B) = (total of all A entries) * (total of all B entries)
    # C[i,k] = 1 * P_0 = (total A) * (total B) for ALL i,k.
    # This is clearly wrong! C[i,k] should be Σ_j A[i,j] B[j,k], not the total product.
    
    # So what's the actual constraint? The constraint must be more specific.
    # The term A[i,j]*B[j,k] should appear in C[i,k] with coefficient exactly 1,
    # and in no other C[i',k'] with nonzero coefficient (or rather, the net coefficient
    # for A[i',j']*B[j',k'] in C[i,k] should be δ_{i,i'}δ_{j,j'}δ_{k,k'}).
    
    # Let me re-derive: C[i,k] = Σ_c W[k*n+i, c] * P_c
    # P_c = (Σ_{i',j'} U[i'*m+j', c] * A[i',j']) * (Σ_{j',k'} V[j'*p+k', c] * B[j',k'])
    #     = Σ_{i',j',j'',k'} U[i'm+j', c] A[i',j'] V[j''p+k', c] B[j'',k']
    #     = Σ_{i',j'',k'} A[i',j''] B[j'',k'] * Σ_{j' = j''} U[i'm+j', c] V[j'p+k', c]  [renaming]
    # Wait no, j' and j'' are independent in the double sum.
    # P_c = Σ_{i'} Σ_{j'} Σ_{j''} Σ_{k'} U[i'm+j', c] V[j''p+k', c] A[i',j'] B[j'',k']
    # For this to give Σ_j A[i,j] B[j,k] in C[i,k], we need the coefficient of A[i',j']B[j'',k'] 
    # in C[i,k] to be δ_{i',i} δ_{j',j''} δ_{k'',k} * 1.
    # 
    # C[i,k] = Σ_c W[k*n+i, c] P_c
    # Coefficient of A[i',j'] B[j'',k''] in C[i,k] is:
    # Σ_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k'', c]
    # This must equal δ_{i',i} * δ_{j',j''} * δ_{k'',k}
    
    # So the constraint is: Σ_c U[i'm+j', c] * V[j''p+k'', c] * W[k''n+i, c] = δ_{i',i} δ_{j',j''} δ_{k'',k}
    # for all i', j', j'', k'', i, k.
    
    # This is much more restrictive! It's not just about T[i,j,k]=1.
    # Let me reparametrize: let a = i', b = j', c' = j'', d = k'', e = i, f = k.
    # Constraint: Σ_c U[a*m+b, c] * V[c'*p+d, c] * W[d*n+e, c] = δ_{a,e} δ_{b,c'} δ_{c',d}  wait no.
    # δ_{i',i} = δ_{a,e}, δ_{j',j''} = δ_{b,c'}, δ_{k'',k} = δ_{d,f}
    
    # So: Σ_c U[a*m+b, c] * V[c'*p+d, c] * W[d*n+e, c] = δ_{a,e} * δ_{b,c'} * δ_{d,f}
    
    # Note that δ_{b,c'} and δ_{d,f} involve different indices. Let me set b = c' (since δ_{b,c'} forces this) and d = f:
    # Then: Σ_c U[a*m+b, c] * V[b*p+d, c] * W[d*n+e, c] = δ_{a,e}
    
    # So for fixed b and d: Σ_c U[a*m+b, c] * V[b*p+d, c] * W[d*n+e, c] = δ_{a,e} for all a, e.
    
    # This means: for each (b,d) pair, the product of the "slabs" of U, V, W gives the identity matrix (n×n).
    
    # More precisely: Let U_b be the n×r matrix with U_b[a, c] = U[a*m+b, c]
    #            V_b,d be the row vector (length r) with V_b,d[c] = V[b*p+d, c]  
    # Wait, V has shape (m*p, r), so V[b*p+d, c] is a single row.
    # Let W_d be the n×r matrix with W_d[e, c] = W[d*n+e, c]
    
    # Then the constraint is: for each b in [0,m), d in [0,p):
    #   Σ_c U_b[a, c] * V[b*p+d, c] * W_d[e, c] = δ_{a,e}
    #   i.e., U_b @ (outer product stuff)... 
    #   Actually: define for each (b,d): 
    #   M_{b,d} = U_b^T * diag(V[b*p+d, :]) * W_d  -- no, that's not right dimensionally.
    
    # Let me think of it as: for fixed b,d, consider the n×n matrix:
    #   Q_{b,d}[a,e] = Σ_c U[a*m+b, c] * V[b*p+d, c] * W[d*n+e, c]
    # This must equal I_n for each (b,d).
    
    # Now U[a*m+b, c] for fixed b is an n×r matrix (rows indexed by a).
    # V[b*p+d, c] for fixed (b,d) is a length-r vector.
    # W[d*n+e, c] for fixed d is an n×r matrix (rows indexed by e).
    
    # Q_{b,d} = U_b * (v_{b,d} element-wise) * W_d^T... let me be careful.
    # Q_{b,d}[a,e] = Σ_c U_b[a,c] * v_{b,d}[c] * W_d[e,c]
    # = Σ_c (U_b[a,c] * v_{b,d}[c]) * W_d[e,c]
    # = [U_b * diag(v_{b,d})] [a,c] * W_d[e,c]  (summing over c)
    # = (U_b * diag(v_{b,d})) @ W_d^T  [a,e]
    
    # So Q_{b,d} = (U_b ⊙ v_{b,d}) @ W_d^T where ⊙ is element-wise multiplication (column-wise).
    # This must equal I_n for all b, d.
    
    # For this to work, we need at least m*p independent conditions (one identity for each (b,d)),
    # each requiring rank at least n in some sense.
    
    # Minimum rank: At least n (since each identity is n×n full rank).
    # But we have m*p = 15 such constraints sharing the same r-dimensional space.
    
    # Lower bound: r ≥ n = 2 (trivial). More useful: consider the "slice" structure.
    # For fixed d, we have m constraints: (U_b ⊙ v_{b,d}) @ W_d^T = I_n for b=0..m-1.
    # This means (U_b ⊙ v_{b,d}) = W_d^{-T} (if W_d is invertible) for all b, which is impossible unless m=1.
    # Actually, (U_b ⊙ v_{b,d}) @ W_d^T = I means U_b ⊙ v_{b,d} = (W_d^T)^{-1} = W_d^{-T} for all b.
    # But U_b ⊙ v_{b,d} depends on both b and d, while W_d^{-T} depends only on d.
    # So for fixed d, U_b ⊙ v_{b,d} must be the same matrix for all b!
    # That means: U_0 ⊙ v_{0,d} = U_1 ⊙ v_{1,d} = ... = U_{m-1} ⊙ v_{m-1,d} = W_d^{-T}
    
    # This is a very strong constraint. It means U_b * v_{b,d} = constant (independent of b) for each d.
    # Similarly for fixed b: (U_b ⊙ v_{b,d}) @ W_d^T = I for all d means U_b ⊙ v_{b,d} = W_d^{-T} for all d.
    # So U_b ⊙ v_{b,d} = W_d^{-T} = U_{b'} ⊙ v_{b',d} for all b, b', d.
    
    # Therefore: U_b ⊙ v_{b,d} is independent of b (for fixed d) AND independent of d (for fixed b).
    # So U_b ⊙ v_{b,d} = S (same matrix for all b,d) where S = W_d^{-T} (which must be independent of d, 
    # so W_d is the same for all d! And S @ W_d^T = I so S = W_d^{-T}).
    
    # So W_d = W for all d (same W matrix), and U_b ⊙ v_{b,d} = W^{-T} for all b,d.
    # This means: U_b ⊙ v_{b,d} is constant. Let's call it C = W^{-T}.
    
    # From U_b ⊙ v_{b,d} = C: v_{b,d}[c] = C[c,c'] / U_b[c,c']... this gets complicated.
    # Actually: U_b[a,c] * v_{b,d}[c] = C[a,c] (since it's an n×r matrix, and C = W^{-T} is n×r).
    # So v_{b,d}[c] = C[a,c] / U_b[a,c] for all a. This requires C[a,c] / U_b[a,c] to be independent of a.
    
    # This is getting very constrained. Let me try a different approach: just search for the minimum rank.
    
    # Given the complexity, let me try to find a known result for 2x3x5 matrix multiplication rank.
    # The rank is at least max(n,m,p) = 5 (Lansky-Schmidt bound: rank ≥ max(n,m,p)).
    # Actually the bound is rank ≥ (n*p*min(m,1))/min(m,1) ... various bounds exist.
    # 
    # For 2x3x5: the trivial rank is 30.
    # 
    # Let me try to construct a rank < 30 solution using the constraints above.
    # 
    # Key insight from the analysis: W_d must be the same for all d (call it W, shape n×r),
    # and U_b ⊙ v_{b,d} = W^{-T} (shape n×r) for all b,d.
    # 
    # From U_b ⊙ v_{b,d} = W^{-T}:
    # (U_b ⊙ v_{b,d})[a,c] = U_b[a,c] * v_{b,d}[c] = (W^{-T})[a,c]
    # 
    # For this to hold for all b (with fixed d):
    # U_b[a,c] * v_{b,d}[c] = (W^{-T})[a,c]
    # v_{b,d}[c] = (W^{-T})[a,c] / U_b[a,c]  for all a
    # 
    # For this to be consistent (independent of a), we need:
    # (W^{-T})[a,c] / U_b[a,c] = (W^{-T})[a',c] / U_b[a',c]  for all a, a', c
    # i.e., U_b[a,c] / U_b[a',c] = (W^{-T})[a,c] / (W^{-T})[a',c]  for all a, a', c
    # i.e., U_b[a,c] / (W^{-T})[a,c] is independent of a.
    # 
    # So for each column c of U_b, the ratio U_b[a,c] / (W^{-T})[a,c] must be constant over a.
    # This means: U_b[:,c] = λ_c * (W^{-T})[:,c]  for some scalar λ_c (depending on b and c).
    
    # Similarly, from the other constraint (fixing b, varying d):
    # v_{b,d}[c] must satisfy: U_b[a,c] * v_{b,d}[c] = (W^{-T})[a,c] for all a
    # So v_{b,d}[c] = (W^{-T})[a,c] / U_b[a,c] = (W^{-T})[a,c] / (λ_c(b) * (W^{-T})[a,c]) = 1/λ_c(b)
    # So v_{b,d}[c] = 1/λ_c(b), which is independent of d!
    
    # So v_{b,d}[c] = 1/λ_c(b) for all d.
    # And U_b[a,c] = λ_c(b) * (W^{-T})[a,c]
    
    # Let me verify: (U_b ⊙ v_{b,d})[a,c] = U_b[a,c] * v_{b,d}[c] = λ_c(b)*(W^{-T})[a,c] * (1/λ_c(b)) = (W^{-T})[a,c]. ✓
    
    # So the structure is:
    # - W (n×r) is a fixed matrix, must be invertible (r ≥ n).
    # - For each b: U_b[:,c] = λ_c(b) * (W^{-T})[:,c] for all a.
    # - For each b: v_{b,d}[c] = 1/λ_c(b) for all d.
    # 
    # So v_{b,d} is independent of d! And U_b depends on b through the vector λ(b) = (λ_0(b), ..., λ_{r-1}(b)).
    # W is fixed.
    
    # Now, are there any additional constraints? Let me check.
    # The original constraint was Σ_c U[a*m+b, c] * V[b*p+d, c] * W[d*n+e, c] = δ_{a,e} δ_{b',b} δ_{d',d}... 
    # Wait, I think I need to recheck. Let me re-derive more carefully.
    
    # The constraint is: for ALL (a, b, b', d, d', e, e'):
    # Σ_c U[a*m+b, c] * V[b'*p+d, c] * W[d*n+e, c] = δ_{a,e'} * δ_{b,b'} * δ_{d,d'}... 
    
    # Hmm, I think I mixed up indices. Let me restart with clear notation.
    # 
    # A is n×m: entries A[i,j] for i∈[0,n), j∈[0,m)
    # B is m×p: entries B[j,k] for j∈[0,m), k∈[0,p)  
    # C is n×p: entries C[i,k] for i∈[0,n), k∈[0,p)
    # C[i,k] = Σ_{j=0}^{m-1} A[i,j] * B[j,k]
    # 
    # Decomposition with rank r:
    # U: (n*m) × r, indexed U[i*m+j, c] = coefficient of A[i,j] in "left factor c"
    # V: (m*p) × r, indexed V[j*p+k, c] = coefficient of B[j,k] in "right factor c"
    # W: (p*n) × r, indexed W[k*n+i, c] = coefficient of product_c in C[i,k]
    # 
    # Computation:
    # P_c = (Σ_{i,j} U[i*m+j, c] * A[i,j]) * (Σ_{j,k} V[j*p+k, c] * B[j,k])
    # C[i,k] = Σ_c W[k*n+i, c] * P_c
    # 
    # Coefficient of A[i0,j0] * B[j1,k1] in C[i,k]:
    # Σ_c U[i0*m+j0, c] * V[j1*p+k1, c] * W[k*n+i, c]
    # 
    # This must equal δ_{j0,j1} (since C[i,k] = Σ_j A[i,j]B[j,k], so A[i0,j0]B[j1,k1] 
    # contributes to C[i,k] iff i0=i, j0=j1, k1=k, and the coefficient is 1).
    # 
    # So: Σ_c U[i0*m+j0, c] * V[j1*p+k1, c] * W[k*n+i, c] = δ_{i0,i} * δ_{j0,j1} * δ_{k1,k}
    
    # Let me set: i0=i (else δ=0, trivially satisfied), k1=k (else δ=0).
    # Remaining constraint: Σ_c U[i*m+j0, c] * V[j1*p+k, c] * W[k*n+i, c] = δ_{j0,j1}
    # for all i, j0, j1, k.
    
    # So for fixed (i, k): Σ_c U[i*m+j0, c] * V[j1*p+k, c] * W[k*n+i, c] = δ_{j0,j1}
    
    # Define: for fixed (i,k), let 
    #   A_{i,k}[j0, c] = U[i*m+j0, c]  (this is an m×r matrix)
    #   B_{i,k}[j1, c] = V[j1*p+k, c]  (m×r matrix)
    #   W_{i,k}[c] = W[k*n+i, c]  (length-r vector)
    # 
    # Then: Σ_c A_{i,k}[j0,c] * B_{i,k}[j1,c] * W_{i,k}[c] = δ_{j0,j1}
    # 
    # This says: A_{i,k} * diag(W_{i,k}) * B_{i,k}^T = I_m
    # (an m×m identity matrix)
    
    # So for each (i,k) pair (there are n*p = 10 of them), we need:
    # A_{i,k} @ diag(W_{i,k}) @ B_{i,k}^T = I_m  (where A_{i,k}, B_{i,k} are m×r slices of U, V)
    
    # A_{i,k} is the m×r submatrix of U with rows {i*m+j : j=0..m-1}
    # B_{i,k} is the m×r submatrix of V with rows {j*p+k : j=0..m-1}
    
    # So for each (i,k): (U_slice_i) @ diag(W[k*n+i, :]) @ (V_slice_k)^T = I_m
    
    # Where U_slice_i = U[i*m : (i+1)*m, :]  (m×r)
    #        V_slice_k = V[k : p : ... wait, V[j*p+k, :] for j=0..m-1.
    #        V_slice_k = V[k:m*p:m, :] -- no, V has rows indexed j*p+k for j in range(m).
    #        So V_slice_k[c_row, :] where c_row = j*p+k, i.e., rows k, p+k, 2p+k, ..., (m-1)p+k.
    #        That's V[k : m*p : p, :] -- actually it's rows [k, k+p, k+2p, ..., k+(m-1)p].
    
    # OK so the constraint is: for each i in [0,n), k in [0,p):
    #   U[i*m:(i+1)*m, :] @ diag(W[k*n+i, :]) @ V[k:m*p:p, :].T = I_m
    
    # This is n*p = 10 constraints, each an m×m = 3×3 identity.
    
    # Now, let me denote:
    # U_i = U[i*m:(i+1)*m, :]  (m×r for each i)
    # V_k = V[k:m*p:p, :]  (m×r for each k, with rows at positions k, k+p, ..., k+(m-1)p)
    # W_{i,k} = W[k*n+i, :]  (length r for each (i,k))
    
    # Constraint: U_i @ diag(W_{i,k}) @ V_k^T = I_m  for all (i,k).
    
    # This means: U_i @ diag(W_{i,k}) = (V_k^T)^{-1} = V_k^{-T}  (assuming V_k is invertible, which requires r ≥ m)
    # So: diag(W_{i,k}) = U_i^{-1} @ V_k^{-T}  (if U_i is invertible, r ≥ m)
    # But diag(W_{i,k}) is a diagonal matrix! So U_i^{-1} @ V_k^{-T} must be diagonal for all (i,k).
    
    # This is a very strong constraint. Let me see if it's satisfiable.
    
    # Alternatively: U_i @ diag(W_{i,k}) @ V_k^T = I means U_i @ diag(W_{i,k}) = V_k^{-T}.
    # Taking transpose: V_k @ diag(W_{i,k}) @ U_i^T = I.
    
    # For this to hold for all i (with fixed k): U_i @ diag(W_{i,k}) = V_k^{-T} for all i.
    # So U_i @ diag(W_{i,k}) is independent of i! 
    # Let's call this M_k = V_k^{-T} (depends only on k).
    # So U_i @ diag(W_{i,k}) = M_k for all i, k.
    
    # Similarly for all k (fixed i): U_i @ diag(W_{i,k}) = M_k, so M_k = U_i^{-1} @ (something)...
    # Actually, from U_i @ diag(W_{i,k}) = V_k^{-T}:
    # For fixed i: U_i @ diag(W_{i,k}) = V_k^{-T} for all k.
    # So diag(W_{i,k}) = U_i^{-1} @ V_k^{-T} for all k.
    # For this to be diagonal: U_i^{-1} @ V_k^{-T} must be diagonal for all k.
    
    # And for fixed k: U_i @ diag(W_{i,k}) = V_k^{-T} for all i.
    # So U_i @ diag(W_{i,k}) is independent of i. Let's call it N_k.
    # N_k = V_k^{-T}.
    # So U_i @ diag(W_{i,k}) = V_k^{-T} for all i.
    # diag(W_{i,k}) = U_i^{-1} @ V_k^{-T} for all i.
    
    # Combining: U_i^{-1} @ V_k^{-T} = U_j^{-1} @ V_k^{-T} for all i,j,k.
    # So U_i^{-1} = U_j^{-1} for all i,j (multiply by V_k^T on the right, assuming V_k^{-T} is invertible).
    # Therefore U_i = U_j for all i! All U slices are the same!
    
    # Similarly, by symmetry, V_k = V_l for all k! All V slices are the same!
    
    # Let U_i = U_s (same m×r matrix for all i) and V_k = V_s (same m×r matrix for all k).
    # Then: U_s @ diag(W_{i,k}) @ V_s^T = I_m for all (i,k).
    # So U_s @ diag(W_{i,k}) = V_s^{-T} for all (i,k).
    # diag(W_{i,k}) = U_s^{-1} @ V_s^{-T} for all (i,k).
    # So W_{i,k} = U_s^{-1} @ V_s^{-T} (diagonal part) for all (i,k).
    # This means W[k*n+i, :] is the same vector for all (i,k)!
    
    # So W is constant: W[k*n+i, c] = w_c for all i, k, c.
    # And the constraint becomes: U_s @ diag(w) @ V_s^T = I_m.
    
    # Where w = (w_0, ..., w_{r-1}).
    
    # So we need: U_s @ diag(w) @ V_s^T = I_m where U_s, V_s are m×r and w is length r.
    
    # This is: Σ_c w_c * U_s[:,c] * V_s[:,c]^T = I_m (sum of r rank-1 matrices = I_m).
    
    # This requires r ≥ m (since I_m has rank m and each term has rank ≤ 1).
    # So r ≥ m = 3.
    
    # But wait, we also need the full matrix W to have shape (p*n, r) = (10, r).
    # If W is constant (all rows the same w), that's fine - W just has r columns and all 10 rows equal w.
    
    # And U has shape (n*m, r) = (6, r) where U[i*m:(i+1)*m, :] = U_s for all i.
    # V has shape (m*p, r) = (15, r) where V[k:m*p:p, :] = V_s for all k.
    # Wait, V[k:m*p:p, :] picks rows k, k+p, k+2p, ..., k+(m-1)p from V.
    # For this to equal V_s for all k, V must be such that V[(i*p+j) : ... ] hmm.
    
    # Let me re-examine V. V has shape (m*p, r) = (15, r).
    # V[j*p+k, c] is the entry at row (j*p+k), column c.
    # V_k = V[k:m*p:p, :] means rows k, k+p, k+2p, ..., k+(m-1)p.
    # For V_k = V_s for all k: V[k, :] = V[k+p, :] = V[k+2p, :] = ... for each residue class k mod p.
    # Wait no: V_k contains rows k, k+p, k+2p, ..., k+(m-1)p.
    # For k=0: rows 0, p, 2p, ..., (m-1)p
    # For k=1: rows 1, p+1, 2p+1, ..., (m-1)p+1
    # etc.
    
    # So V[0,:], V[p,:], V[2p,:], ..., V[(m-1)p,:] must all equal the first row of V_s.
    # V[1,:], V[p+1,:], ..., V[(m-1)p+1,:] must all equal the second row of V_s.
    # etc.
    
    # So V is determined by V_s: V[j*p+k, :] = V_s[j, :] for all j, k.
    # Similarly, U[i*m+j, :] = U_s[j, :] for all i, j.
    
    # So the decomposition is fully parameterized by:
    # - U_s: m×r matrix
    # - V_s: m×r matrix  
    # - w: length-r vector
    # with constraint: U_s @ diag(w) @ V_s^T = I_m
    
    # And the full matrices are:
    # U[i*m+j, c] = U_s[j, c] for all i, j, c
    # V[j*p+k, c] = V_s[j, c] for all j, k, c
    # W[k*n+i, c] = w[c] for all i, k, c
    
    # The rank is r. We need r ≥ m = 3 (for the identity decomposition).
    
    # Can we achieve r = 3? We need: U_s @ diag(w) @ V_s^T = I_3 where U_s, V_s are 3×3 and w is length 3.
    # U_s @ diag(w) is a 3×3 matrix (columns of U_s scaled by w).
    # Let M = U_s @ diag(w) (3×3). Then M @ V_s^T = I_3, so V_s^T = M^{-1}, i.e., V_s = M^{-T}.
    # M can be any invertible 3×3 matrix. So yes, r = 3 is achievable!
    
    # Example: U_s = I_3, w = (1,1,1), V_s = I_3.
    # Check: I_3 @ diag(1,1,1) @ I_3^T = I_3. ✓
    
    # So the full decomposition with r=3:
    # U[0,:] = U[1,:] = ... = U[5,:] -- wait, U has 6 rows (n*m=6).
    # U[0,:] = U_s[0,:] = [1,0,0]
    # U[1,:] = U_s[1,:] = [0,1,0]
    # U[2,:] = U_s[2,:] = [0,0,1]
    # U[3,:] = U_s[0,:] = [1,0,0]  (since i=1, j=0)
    # U[4,:] = U_s[1,:] = [0,1,0]
    # U[5,:] = U_s[2,:] = [0,0,1]
    
    # V has 15 rows (m*p=15).
    # V[j*p+k, :] = V_s[j, :] for all j,k.
    # V[0,:] = V[5,:] = V[10,:] = V_s[0,:] = [1,0,0]
    # V[1,:] = V[6,:] = V[11,:] = V_s[1,:] = [0,1,0]
    # V[2,:] = V[7,:] = V[12,:] = V_s[2,:] = [0,0,1]
    # V[3,:] = V[8,:] = V[13,:] = V_s[0,:] = [1,0,0]
    # V[4,:] = V[9,:] = V[14,:] = V_s[1,:] = [0,1,0]
    # Wait, j goes 0,1,2 and k goes 0,1,2,3,4.
    # Row index = j*p + k = j*5 + k.
    # j=0: rows 0,1,2,3,4 → V_s[0]
    # j=1: rows 5,6,7,8,9 → V_s[1]
    # j=2: rows 10,11,12,13,14 → V_s[2]
    
    # V[0,:] = [1,0,0], V[1,:] = [1,0,0], V[2,:] = [1,0,0], V[3,:] = [1,0,0], V[4,:] = [1,0,0]
    # V[5,:] = [0,1,0], V[6,:] = [0,1,0], V[7,:] = [0,1,0], V[8,:] = [0,1,0], V[9,:] = [0,1,0]
    # V[10,:] = [0,0,1], V[11,:] = [0,0,1], V[12,:] = [0,0,1], V[13,:] = [0,0,1], V[14,:] = [0,0,1]
    
    # W has 10 rows (p*n=10). W[k*n+i, c] = w[c] = [1,1,1] for all i,k.
    
    # Let me verify: C[i,k] should be Σ_j A[i,j]*B[j,k].
    # P_c = (Σ_{i,j} U[i*m+j,c] A[i,j]) * (Σ_{j,k} V[j*p+k,c] B[j,k])
    # P_0 = (A[0,0]+A[1,0]) * (B[0,0]+B[0,1]+B[0,2]+B[0,3]+B[0,4])
    # P_1 = (A[0,1]+A[1,1]) * (B[1,0]+B[1,1]+B[1,2]+B[1,3]+B[1,4])
    # P_2 = (A[0,2]+A[1,2]) * (B[2,0]+B[2,1]+B[2,2]+B[2,3]+B[2,4])
    # C[i,k] = P_0 + P_1 + P_2
    # 
    # But C[0,0] should be A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # And our formula gives C[0,0] = (A[0,0]+A[1,0])(B[0,0]+...+B[0,4]) + (A[0,1]+A[1,1])(B[1,0]+...+B[1,4]) + ...
    # This is NOT equal to A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] in general!
    
    # So something is wrong with my derivation. Let me recheck.
    
    # Ah, I think I made an error. The constraint Σ_c U[i*m+j0, c] * V[j1*p+k, c] * W[k*n+i, c] = δ_{j0,j1}
    # is necessary and sufficient for the coefficient of A[i,j0]*B[j1,k] in C[i,k] to be δ_{j0,j1}.
    # But C[i,k] = Σ_c W[k*n+i,c] * P_c, and P_c involves ALL entries of A and B, not just A[i,·] and B[·,k].
    
    # Let me recompute the coefficient of A[i0,j0]*B[j1,k1] in C[i,k] more carefully.
    # P_c = (Σ_{i',j'} U[i'*m+j', c] * A[i',j']) * (Σ_{j'',k'} V[j''*p+k', c] * B[j'',k'])
    #     = Σ_{i',j',j'',k'} U[i'm+j', c] V[j''p+k', c] A[i',j'] B[j'',k']
    # C[i,k] = Σ_c W[k*n+i, c] P_c
    #         = Σ_c Σ_{i',j',j'',k'} W[k*n+i, c] U[i'm+j', c] V[j''p+k', c] A[i',j'] B[j'',k']
    # 
    # Coefficient of A[i0,j0]*B[j1,k1] in C[i,k]:
    # Σ_c W[k*n+i, c] U[i0*m+j0, c] V[j1*p+k1, c]
    # 
    # This must equal the coefficient of A[i0,j0]*B[j1,k1] in the true product C[i,k] = Σ_j A[i,j]B[j,k].
    # A[i0,j0]*B[j1,k1] appears in C[i,k] iff i0=i, j0=j1, k1=k. Coefficient is 1.
    # Otherwise coefficient is 0.
    
    # So: Σ_c U[i0*m+j0, c] V[j1*p+k1, c] W[k*n+i, c] = δ_{i0,i} δ_{j0,j1} δ_{k1,k}
    
    # This is what I had before. And I derived that this forces U_i, V_k, W_{i,k} to have the 
    # specific structure I found, which leads to the "constant W" conclusion.
    
    # But then the verification failed! Let me check where I went wrong.
    
    # With U[i*m+j,c] = U_s[j,c], V[j*p+k,c] = V_s[j,c], W[k*n+i,c] = w_c:
    # Coefficient of A[i0,j0]*B[j1,k1] in C[i,k]:
    # Σ_c U_s[j0,c] * V_s[j1,c] * w_c
    # 
    # This equals δ_{i0,i}δ_{j0,j1}δ_{k1,k} ONLY if it doesn't depend on i0, k1, i, k (other than through the deltas).
    # But it depends on j0 and j1 only, not on i0, i, k1, k.
    # For it to equal δ_{i0,i}δ_{j0,j1}δ_{k1,k}, we need:
    # - When i0≠i or k1≠k: the sum must be 0. But the sum only depends on j0,j1, so it can't be 
    #   0 for some (i0,i) and nonzero for others (with same j0,j1). Contradiction unless the sum is always 0,
    #   which contradicts the case i0=i, j0=j1, k1=k where it must be 1.
    
    # So my derivation must have an error. Let me recheck.
    
    # The error: I said "for fixed (i,k): Σ_c U[i*m+j0,c] V[j1*p+k,c] W[k*n+i,c] = δ_{j0,j1}"
    # This is correct. But then I defined U_i as the m×r submatrix and wrote U_i @ diag(W_{i,k}) @ V_k^T = I_m.
    # Let me recheck: (U_i @ diag(W_{i,k}) @ V_k^T)[j0, j1] = Σ_c U_i[j0,c] * W_{i,k}[c] * V_k[j1,c]
    # = Σ_c U[i*m+j0,c] * W[k*n+i,c] * V[j1*p+k,c] = δ_{j0,j1} ✓
    
    # So the constraint IS: U_i @ diag(W_{i,k}) @ V_k^T = I_m for all (i,k).
    
    # And I derived U_i = U_s for all i, V_k = V_s for all k, W_{i,k} = w for all (i,k).
    # But then the full coefficient becomes Σ_c U_s[j0,c] V_s[j1,c] w_c, independent of i0, i, k1, k.
    # This CANNOT equal δ_{i0,i}δ_{j0,j1}δ_{k1,k} in general.
    
    # So where's the error in the derivation that U_i = U_s?
    
    # Let me recheck: "For fixed k: U_i @ diag(W_{i,k}) = V_k^{-T} for all i."
    # This comes from U_i @ diag(W_{i,k}) @ V_k^T = I → U_i @ diag(W_{i,k}) = V_k^{-T}. ✓
    # "So U_i @ diag(W_{i,k}) is independent of i." ✓ (it's always V_k^{-T})
    
    # "Taking this for all i: U_i @ diag(W_{i,k}) = U_j @ diag(W_{j,k}) = V_k^{-T}"
    # So U_i @ diag(W_{i,k}) = U_j @ diag(W_{j,k}) for all i,j. This does NOT mean U_i = U_j!
    # It means the products are equal, but U_i and W_{i,k} can differ.
    
    # I then said "diag(W_{i,k}) = U_i^{-1} @ V_k^{-T} for all i"
    # And "U_i^{-1} @ V_k^{-T} = U_j^{-1} @ V_k^{-T} for all i,j"
    # This requires V_k^{-T} to be "full" in some sense. Multiplying by V_k^T: U_i^{-1} = U_j^{-1}. ✓
    # Wait, that IS valid! U_i^{-1} @ V_k^{-T} = U_j^{-1} @ V_k^{-T}
    # → (U_i^{-1} - U_j^{-1}) @ V_k^{-T} = 0
    # → U_i^{-1} = U_j^{-1} (since V_k^{-T} is invertible, it has a right inverse)
    # → U_i = U_j.
    
    # So the derivation seems correct, but the conclusion contradicts the verification. 
    # Unless I'm wrong that V_k^{-T} is invertible. V_k is m×r. For V_k to be invertible as a square matrix, we need r = m. But in general r ≥ m and V_k is m×r (wide), so V_k^{-T} doesn't exist in the usual sense!
    
    # AH, that's the error! V_k is m×r, not square (unless r=m). So V_k^T is r×m, and 
    # U_i @ diag(W_{i,k}) is m×r, and U_i @ diag(W_{i,k}) @ V_k^T is m×m = I.
    # But we can't "invert" V_k^T because it's r×m with r ≥ m.
    
    # The correct step: U_i @ diag(W_{i,k}) @ V_k^T = I_m.
    # This means U_i @ diag(W_{i,k}) is a left inverse of V_k^T.
    # There are many left inverses. So my previous argument doesn't work when r > m.
    
    # For r = m: all matrices are square, and the argument goes through, giving the constant-W structure
    # which doesn't work. So r = m = 3 doesn't work via this approach?
    
    # Wait, let me just verify directly. With r=3, U_s=V_s=I_3, w=(1,1,1):
    # Coefficient of A[i0,j0]*B[j1,k1] in C[i,k] = Σ_c δ_{j0,c} δ_{j1,c} * 1 = δ_{j0,j1}.
    # But we need δ_{i0,i}δ_{j0,j1}δ_{k1,k}. So this gives δ_{j0,j1} which is 1 when j0=j1 regardless of i0,i,k1,k.
    # That's wrong. So r=3 with this structure doesn't work.
    
    # The issue is that the constraint U_i @ diag(W_{i,k}) @ V_k^T = I_m for all (i,k) is NOT sufficient!
    # I only enforced the coefficient condition for the case where i0=i and k1=k (fixing i,k on the output side).
    # But I also need the coefficient to be 0 when i0≠i or k1≠k, which imposes ADDITIONAL constraints 
    # that mix different (i,k) pairs!
    
    # Let me re-examine. The full constraint is:
    # Σ_c U[i0*m+j0,c] V[j1*p+k1,c] W[k*n+i,c] = δ_{i0,i} δ_{j0,j1} δ_{k1,k}
    # for ALL i0,j0,j1,k1,i,k.
    
    # Fix j0,j1 and set j0=j1 (else δ=0, trivial). Then for all i0,k1,i,k:
    # Σ_c U[i0*m+j0,c] V[j0*p+k1,c] W[k*n+i,c] = δ_{i0,i} δ_{k1,k}
    
    # This is an n×p matrix (indexed by (k1,i0) on one side and (i,k) on the other) that must equal I_{n×p}... 
    # wait, it's n×n×p×p. Let me think of it as: for fixed j0:
    # M_{j0}[i0, k1, i, k] = Σ_c U[i0*m+j0,c] V[j0*p+k1,c] W[k*n+i,c] = δ_{i0,i} δ_{k1,k}
    
    # This is a 4-index tensor that must factor as δ_{i0,i} * δ_{k1,k}.
    
    # Let me define (for fixed j0):
    # A_{j0}[i0, c] = U[i0*m+j0, c]  (n×r)
    # B_{j0}[k1, c] = V[j0*p+k1, c]  (p×r)
    # C[c] = ... hmm, W[k*n+i,c] couples k and i.
    
    # M_{j0}[i0,k1,i,k] = Σ_c A_{j0}[i0,c] B_{j0}[k1,c] W[k*n+i,c]
    
    # Let W be a (p×n) matrix reshaped: W[k*n+i, c]. Define W_mat[c, k, i] = W[k*n+i, c].
    # Then M_{j0}[i0,k1,i,k] = Σ_c A_{j0}[i0,c] B_{j0}[k1,c] W_mat[c,k,i]
    
    # We need: M_{j0} = δ_{i0,i} * δ_{k1,k} as a 4D tensor.
    
    # This is equivalent to: the tensor M_{j0} has the form of a "product" of two Kronecker deltas.
    
    # This is a very strong constraint. In tensor language, M_{j0} must be in the image of the map:
    # (i0,k1) × (i,k) ↦ δ_{i0,i}δ_{k1,k}
    
    # One way to satisfy this: if W_mat[c,k,i] = W_c * δ-like structure... 
    
    # Actually, let me think about it as: for the multiplication to work, we need the "routing" to be correct.
    # The term A[i0,j0]*B[j0,k1] should ONLY contribute to C[i0,k1], not to any other C[i,k].
    
    # This means the c-th intermediate product P_c, when multiplied by W[k*n+i,c], should route 
    # A[i0,j0]*B[j0,k1] to C[i0,k1] and nowhere else.
    
    # I think the cleanest way to handle this is to just do a numerical search for the minimum rank.
    # Given the constraints, let me try rank 6, 8, 10, etc. and see if a solution exists.
    
    # Actually, let me try a known approach: for matrix multiplication of n×m by m×p,
    # the rank is at most n*m*p (schoolbook) and at least max(n,m,p) (trivial bound) or 
    # at least (n*p*m)^(1/2) (geometric mean bound, I think).
    
    # For 2×3×5: max(n,m,p)=5, geometric mean=(2*3*5)^(1/3)≈3.1, so rank ≥ 5.
    
    # Let me try to see if rank 5, 6, 7, 8, ... is achievable by random search or structured construction.
    
    # Given time constraints, let me try a randomized approach to find a low-rank decomposition.
    
    # Actually, let me just try the obvious: is rank 30 the minimum? Probably not.
    # Let me try rank = n*m = 6 or m*p = 15 or p*n = 10 or some other value.
    
    # Let me try a structured approach with rank = m*p = 15:
    # Use one coefficient per (j,k) pair. U[i*m+j, c] where c = j*p+k.
    # U[i*m+j, j'*p+k'] = δ_{j,j'} (independent of i!) -- no, that breaks things.
    
    # Let me just do the schoolbook rank 30 solution. It's guaranteed correct.
    # Actually wait - the task asks for MINIMAL rank. Let me try to find something better.
    
    # Let me try a simple optimization: start with random matrices and use gradient descent to minimize
    # the constraint violation, then round to rationals.
    
    # Given time constraints, let me try a few structured constructions and verify them.
    
    # Construction attempt 1: rank = n*p = 10.
    # Idea: group by (i,k) output index. For each (i,k), have coefficients that compute 
    # Σ_j A[i,j]B[j,k] directly.
    # U[i*m+j, c_{i,k}] = 1 if c = c_{i,k} (one per i), 0 otherwise -- but each (i,k) needs its own c.
    # That gives n*p = 10 columns.
    # U[i*m+j, c_{i',k'}] = δ_{i,i'} (contribution from row i of A to output row i)
    # V[j*p+k, c_{i,k']} = δ_{k,k'} (contribution from col k of B to output col k)
    # W[k*n+i, c_{i',k'}] = δ_{(i,k),(i',k')}
    
    # Let's check: coefficient of A[i0,j0]*B[j1,k1] in C[i,k]:
    # Σ_c U[i0*m+j0,c] V[j1*p+k1,c] W[k*n+i,c]
    # U[i0*m+j0, c_{i',k'}] = δ_{i0,i'}
    # V[j1*p+k1, c_{i,k''}] = δ_{k1,k''}
    # W[k*n+i, c_{i',k''}] = δ_{i,i'} δ_{k,k''}
    
    # Sum over c: Σ_{i',k''} δ_{i0,i'} δ_{k1,k''} δ_{i,i'} δ_{k,k''}
    # = Σ_{i',k''} δ_{i0,i'} δ_{i,i'} δ_{k1,k''} δ_{k,k''}
    # = δ_{i0,i} δ_{k1,k} (since i'=i0=i and k''=k1=k must both hold)
    
    # But we need δ_{i0,i} δ_{j0,j1} δ_{k1,k}. The j0,j1 condition is missing!
    # The sum doesn't depend on j0 or j1 at all (U only depends on i0, V only on k1, W only on k,i).
    # So this gives δ_{i0,i}δ_{k1,k} for all j0,j1, but we need it to also have δ_{j0,j1}.
    # So this construction doesn't work.
    
    # The issue: we need the j-index to be matched between A and B. 
    # The coefficient must "know" that j0=j1.
    
    # This suggests we need at least one degree of freedom per (j) value in the intermediate.
    
    # Let me try rank = n*m*p but with some sharing. Or let me just do a proper search.
    
    # Given the time pressure, let me implement the schoolbook solution (rank 30) which is known correct,
    # and then try to improve it with a quick search if time permits.
    
    rank = n * m * p  # 30
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    result = {"U": U, "V": V, "W": W}
    
    # Try to improve: attempt rank 15, 10, 8, 6, 5 with random initialization and verification
    def verify(U, V, W, n, m, p, tol=1e-8):
        """Check if the decomposition is valid."""
        rank = len(U[0])
        # For each (i0, j0, j1, k1, i, k), check the constraint
        # Sum over c of U[i0*m+j0,c] * V[j1*p+k1,c] * W[k*n+i,c] = delta(i0,i)*delta(j0,j1)*delta(k1,k)
        for i0 in range(n):
            for j0 in range(m):
                for j1 in range(m):
                    for k1 in range(p):
                        for i in range(n):
                            for k in range(p):
                                expected = (1 if i0==i and j0==j1 and k1==k else 0)
                                actual = sum(U[i0*m+j0][c] * V[j1*p+k1][c] * W[k*n+i][c] for c in range(rank))
                                if abs(actual - expected) > tol:
                                    return False
        return True
    
    # The full verification is O(n^2 * m^2 * p^2 * r) which for n=2,m=3,p=5,r=30 is 
    # 4*9*25*30 = 27000 operations - fast enough.
    
    deadline = 130  # seconds
    best_rank = rank
    best_result = result
    
    # Try to find lower rank solutions via random search
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    for target_rank in [15, 12, 10, 9, 8, 7, 6, 5]:
        if time.time() > deadline:
            break
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Random integer matrices
            U_r = np.zeros((N, target_rank), dtype=int)
            V_r = np.zeros((M, target_rank), dtype=int)
            W_r = np.zeros((P, target_rank), dtype=int)
            for c in range(target_rank):
                U_r[:, c] = np.random.randint(-2, 3, size=N)
                V_r[:, c] = np.random.randint(-2, 3, size=M)
                W_r[:, c] = np.random.randint(-2, 3, size=P)
            
            # Check
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            if verify(U_list, V_list, W_list, n, m, p, tol=1e-10):
                best_rank = target_rank
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
    
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
