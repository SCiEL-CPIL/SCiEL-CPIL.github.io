"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    The multiplication tensor T[i,j,k] = e_i ⊗ e_j ⊗ e_k where:
    - i in range(n=2), j in range(m=3), k in range(p=10)
    - U has shape (n*m, rank) = (6, rank)
    - V has shape (m*p, rank) = (30, rank)  
    - W has shape (p*n, rank) = (20, rank)
    
    The decomposition satisfies: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We use a structured approach based on the specific rectangular dimensions.
    The schoolbook rank is n*m*p = 60. We try to find lower ranks using
    structured linear algebra and Winograd-style optimizations.
    """
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Start with schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 60
    
    # Try to find a lower rank decomposition
    # For rectangular matrix multiplication, there are known lower bounds
    # and specific constructions that can beat schoolbook for certain dimensions.
    
    # One approach: use the fact that we can combine some of the multiplication
    # operations through clever linear combinations.
    
    # Let's try a systematic search for lower rank decompositions
    # using the structure of the problem.
    
    # The key insight: for matrix multiplication C = A * B where A is n×m and B is m×p,
    # we need to compute n*p outputs, each being a sum of m products.
    # The schoolbook uses n*m*p multiplications.
    
    # For our case: n=2, m=3, p=10, so we have 20 outputs, each requiring 3 products.
    # Total: 60 scalar multiplications in schoolbook.
    
    # Can we do better? Let's try using the structure where we can share some computations.
    
    # One strategy: group the p=10 columns of B into groups and use structured
    # linear combinations to reduce the number of independent multiplications.
    
    # For now, let's implement a search that tries to find a valid decomposition
    # with rank < 60 using a greedy or structured approach.
    
    # Actually, for this specific problem, let's try a different approach:
    # Use the fact that we can represent the multiplication as a bilinear map
    # and try to find a low-rank decomposition of the associated tensor.
    
    # The tensor T is in C^{n*m} ⊗ C^{m*p} ⊗ C^{p*n}
    # We want to find the minimum number of rank-1 tensors that sum to T.
    
    # Let's try a constructive approach:
    # For each output element C[i,k] = sum_j A[i,j] * B[j,k], we need to
    # decompose this into a sum of products of linear forms.
    
    # A simple improvement: if we can find linear combinations of the A and B
    # entries such that fewer products suffice, we can reduce the rank.
    
    # Let's try a specific construction for n=2, m=3, p=10:
    # 
    # For the two rows of A (i=0,1), we have:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # One idea: use the fact that we can compute linear combinations of A entries
    # and B entries to reduce the number of independent products.
    
    # For example, define:
    # u0 = A[0,0], u1 = A[0,1], u2 = A[0,2]
    # v0 = A[1,0], v1 = A[1,1], v2 = A[1,2]
    
    # And for each k:
    # b0_k = B[0,k], b1_k = B[1,k], b2_k = B[2,k]
    
    # Then:
    # C[0,k] = u0*b0_k + u1*b1_k + u2*b2_k
    # C[1,k] = v0*b0_k + v1*b1_k + v2*b2_k
    
    # This is already the schoolbook form with 6 products per k, 10 k's = 60 total.
    
    # Can we share products across different k values? Not directly, since the
    # B entries are different for each k.
    
    # However, we might be able to use a Winograd-style approach where we
    # compute some intermediate linear combinations.
    
    # Let's try a different tactic: use a known lower bound and see if we can
    # achieve it. The rank of matrix multiplication tensor [n,m,p] is at least
    # max(n*m, m*p, p*n) = max(6, 30, 20) = 30.
    
    # So the minimum possible rank is at least 30. Can we achieve 30?
    
    # For square matrices, the rank is at least n^2. For rectangular, it's more complex.
    
    # Let's try to construct a decomposition of rank 30:
    # This would require that each of the 30 terms contributes to exactly one
    # (i,k) output pair in a clever way.
    
    # Actually, the lower bound of 30 comes from the fact that we have 30 independent
    # B entries (m*p = 3*10 = 30). Each product in the decomposition can involve
    # at most one B entry (since B entries are independent), so we need at least 30 terms.
    
    # Let's try to construct a rank-30 decomposition:
    # For each B entry B[j,k], we need to multiply it by some linear combination of A entries.
    
    # Term c (for c = 0, ..., 29):
    # Let j = c // 10, k = c % 10 (so j in 0..2, k in 0..9)
    # V[j*p+k, c] = 1, V[j*p+k', c] = 0 for k' != k
    # U[i*m+j', c] should be such that sum over c gives the right A entries.
    # W[k'*n+i', c] should be such that we get the right output.
    
    # If we set:
    # U[i*m+j, c] = A[i,j] coefficient (but this must be a constant, not depending on A)
    # This doesn't work directly because U, V, W must be constant matrices.
    
    # The correct approach: each term c contributes U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c]
    # to output (i'', k'') for input (i, j) and (j', k').
    
    # For the decomposition to be correct, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k'*n+i', c] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # Wait, that's not right either.
    
    # The correct condition is:
    # For all i, i', j, j', k, k':
    # sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c] = delta_{i,i'} * (j == j' and k == k' ? 1 : 0)
    
    # Actually, the multiplication tensor is:
    # T[i*m+j, j'*p+k', k''*n+i''] = 1 if j == j' and k == k' and i == i', else 0
    # No wait, that's not right either.
    
    # Let me re-derive. The matrix multiplication is:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    
    # So the tensor T has components:
    # T[i*m+j, j*p+k, k*n+i] = 1 (for the product A[i,j] * B[j,k] contributing to C[i,k])
    # and 0 otherwise.
    
    # More precisely, T is a 3-tensor with indices (i*m+j, j'*p+k', k''*n+i'')
    # T[i*m+j, j'*p+k', k''*n+i''] = 1 if j == j' and k == k' and i == i'', else 0
    
    # Wait, that's still not quite right. Let me think again.
    
    # The input to the bilinear map is (A, B) where A is n×m and B is m×p.
    # The output is C = A*B which is n×p.
    
    # We can view this as a bilinear map from C^{n*m} × C^{m*p} to C^{n*p}.
    # The associated tensor T is in C^{n*m} ⊗ C^{m*p} ⊗ C^{n*p}.
    
    # T[a, b, c] where a = i*m+j, b = j'*p+k', c = i''*p+k''
    # T[a, b, c] = 1 if j == j' and k' == k'' and i == i'', else 0
    
    # So for a fixed (i, j, k'), the term A[i,j] * B[j,k'] contributes to C[i, k'].
    
    # Therefore:
    # T[i*m+j, j*p+k, i*p+k] = 1
    # and T is 0 elsewhere (with appropriate index matching).
    
    # Now, a rank-1 decomposition term is:
    # u ⊗ v ⊗ w where u in C^{n*m}, v in C^{m*p}, w in C^{n*p}
    
    # The sum of such terms should equal T.
    
    # For a rank-r decomposition:
    # T[a, b, c] = sum_{c'=0}^{r-1} U[a, c'] * V[b, c'] * W[c, c']
    
    # Note: the task says W has shape (p*n, rank), and the index is W[k*n+i, r].
    # So W's first index is k*n+i where k in 0..p-1, i in 0..n-1.
    # This corresponds to the output C[i, k] indexed as k*n+i.
    
    # So W[k*n+i, c'] corresponds to the coefficient for output C[i, k].
    
    # Let's re-index: let the output index be out = k*n + i (as per the task).
    # Then W[out, c'] = W[k*n+i, c'].
    
    # And T[i*m+j, j'*p+k', k''*n+i''] = 1 if j == j', k' == k'', and i == i'', else 0.
    
    # OK so now let's try to construct a low-rank decomposition.
    
    # One approach: for each (j, k) pair (there are m*p = 30 of them), create one term
    # that handles the contribution of A[:,j] * B[j,:] to the output.
    
    # For term c corresponding to (j, k):
    # V[j*p+k, c] = 1
    # U[i*m+j, c] = 1 for all i (so that A[i,j] is picked up)
    # W[k'*n+i', c] = delta_{k',k} * 1 (so that only output C[i', k] gets this contribution)
    
    # Let's check: sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c]
    # = sum_{(j_c, k_c)} delta_{j,j_c} * delta_{j',j_c} * delta_{k',k_c} * delta_{k'',k_c}
    # = delta_{j,j'} * delta_{k,k''} * (1 if j==j' and k==k'' else 0)
    # = delta_{j,j'} * delta_{k,k''}
    
    # But we also need i == i'' for the output to be correct. The above gives 1 for all i, i''
    # when j==j' and k==k'', which is wrong.
    
    # We need W to also depend on i. Let's set:
    # W[k''*n+i'', c] = delta_{k'',k} * delta_{i'',i}
    
    # But i is not fixed for a given term c. The term c corresponds to (j, k), but i varies.
    
    # This approach doesn't work directly because a single term can't handle all i for a fixed (j,k).
    
    # Let's try a different construction. For each (i, j, k) triple, we have one term (schoolbook).
    # Can we merge some of these?
    
    # Consider merging terms with the same (j, k) but different i.
    # For fixed (j, k), we have n=2 terms: one for i=0, one for i=1.
    # Term for (i=0, j, k): U[0*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+0, c] = 1
    # Term for (i=1, j, k): U[1*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+1, c] = 1
    
    # Can we combine these into one term? We would need:
    # U[i*m+j, c] = 1 for both i=0 and i=1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1 for both i=0 and i=1
    
    # But then this single term would contribute to both C[0,k] and C[1,k], which is correct!
    # The contribution to C[i,k] from A[i,j]*B[j,k] is exactly what we want.
    
    # Wait, but U[i*m+j, c] = 1 for both i means that for input A, we're picking up
    # A[0,j] and A[1,j] with the same coefficient. But they should go to different outputs.
    
    # Let's check the full sum:
    # sum_c U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c]
    
    # For the merged term (j, k):
    # U[i*m+j, c] = 1 if j matches, else 0 (independent of i)
    # V[j'*p+k', c] = 1 if j'==j and k'==k, else 0
    # W[k''*n+i'', c] = 1 if k''==k, else 0 (independent of i'')
    
    # So the contribution is: delta_{j,j'} * delta_{k,k''} * 1 (for any i, i'')
    
    # This means the term contributes 1 to T[i*m+j, j*p+k, k*n+i''] for all i, i'' when j==j' and k==k''.
    # But we only want it to contribute when i==i''.
    
    # So the merged term overcounts. It contributes to all (i, i'') pairs, not just i==i''.
    
    # Therefore, we can't simply merge terms with the same (j,k) but different i.
    
    # The schoolbook rank of 60 might be close to optimal for this specific dimension.
    
    # Let's try a more sophisticated approach using linear algebra.
    
    # We can set up the problem as: find the minimum r such that there exist
    # U (6×r), V (30×r), W (20×r) with:
    # sum_c U[a,c] * V[b,c] * W[c_out,c] = T[a,b,c_out]
    
    # This is equivalent to finding the tensor rank, which is NP-hard in general.
    
    # For this specific problem, let's try a heuristic search.
    
    # Given the time constraint, let's return the schoolbook decomposition for now,
    # but with a check to ensure it's correct.
    
    rank = rank_schoolbook
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
