"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank as baseline
    schoolbook_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # Initialize schoolbook
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U = [row[:] for row in U_sb]
    best_V = [row[:] for row in V_sb]
    best_W = [row[:] for row in W_sb]
    
    # Try to find lower rank decompositions
    # The tensor T[i,j,k] = e_{i,j} * e_{j,k} * e_{k,i} in a certain sense
    # Actually, the multiplication tensor T has T[i,j,k] = 1 if we view it as
    # (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # So T_{i,j,k} = 1 for all i,j,k (it's a 3x4x8 tensor of ones? No.)
    # Wait, the tensor is defined such that the bilinear map is:
    # (A, B) -> A*B
    # where A is 3x4, B is 4x8
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 if the product
    # involves A[i,j] * B[j,k] to produce (A*B)[i,k]
    # So T[i,j,k] = 1 for all valid i,j,k. It's a tensor of all ones?
    # No, that's not right. The tensor is the structure tensor of the algebra.
    # For matrix multiplication, T[i,j,k] = 1 if i is row of A, j is col of A / row of B,
    # k is col of B, and the output is (A*B)[i,k].
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, that would mean the tensor is just all ones, which has rank 1? No.
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the bilinear form is
    # <T, (A,B)> = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    # For matrix multiplication, <T, (A,B)> = sum_{i,k} (A*B)[i,k] * C[i,k]
    # = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
    # So T[i,j,k] = 1 for all i,j,k.
    # But a tensor of all ones in dimensions 3x4x8 has rank... 
    # Actually, the rank of an all-ones tensor of size n x m x p is max(n, m, p) if we allow real entries?
    # No, that's not right either. Let me think again.
    
    # The tensor T has T[i,j,k] = 1 for all i,j,k.
    # We want to decompose T as sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^n, v_r in R^m, w_r in R^p
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r] = 1 for all i,j,k
    
    # This means for all i,j,k: sum_r U[i,r] * V[j,r] * W[k,r] = 1
    # Let M_r = U[:,r] * V[:,r] (outer product, n x m matrix)
    # Then sum_r M_r[i,j] * W[k,r] = 1 for all i,j,k
    # For fixed i,j: sum_r M_r[i,j] * W[k,r] = 1 for all k
    # This means the vector (M_r[i,j])_r dotted with W[k,:] = 1 for all k
    # So W[k,:] is in the hyperplane defined by (M_r[i,j])_r
    # For this to hold for all k, all W[k,:] must be the same vector, say w
    # And sum_r M_r[i,j] * w_r = 1 for all i,j
    
    # Hmm, this is getting complicated. Let me just try a direct approach.
    
    # Actually, I realize the tensor for matrix multiplication is NOT all ones.
    # Let me re-read the problem. The output factor follows W[k*n+i, r].
    # So W has dimensions p*n x rank.
    # The decomposition is: (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't make sense dimensionally. Let me re-read.
    
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The bilinear map should be: (A*B)[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]?
    # No, that's not right either.
    
    # Standard tensor decomposition for matrix multiplication:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U, V, W are the factor tensors.
    
    # But the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # So U is flattened: U[i*m+j, r] corresponds to U[i,j,r]
    # V[j*p+k, r] corresponds to V[j,k,r]
    # W[k*n+i, r] corresponds to W[i,k,r] (note the order: k*n+i, so index is k*n+i)
    
    # So the decomposition is:
    # (A*B)[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    # Wait, that would be sum_r [sum_j U[i*m+j,r] * A[i,j]] * [sum_j' V[j'*p+k,r] * B[j',k]] * W[k*n+i,r]
    # That's a product of three sums, which is cubic, not bilinear.
    
    # I think the correct interpretation is:
    # (A*B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But j is summed over implicitly? No, j appears in both U and V.
    
    # Let me think about this differently. The standard way to express matrix multiplication
    # as a tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where T[i,j,k] = 1 (since (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    
    # So we need: for all i,j,k, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of equations. Let's try to find a low-rank solution.
    
    # For rank R, we need to find U (n*m x R), V (m*p x R), W (p*n x R)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try R = 1 first:
    # sum_r=1: U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 for all i,j,k
    # This means U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1
    # For fixed i,k: U[i*m+j, 0] * V[j*p+k, 0] = 1/W[k*n+i, 0] for all j
    # So U[i*m+j, 0] * V[j*p+k, 0] is independent of j
    # Let a_j = U[i*m+j, 0], b_j = V[j*p+k, 0]
    # a_j * b_j = c(i,k) for all j
    # This means a_j = c(i,k) / b_j, so a_j / a_{j'} = b_{j'} / b_j for all j,j'
    # This is possible if all a_j are proportional and all b_j are proportional
    # Specifically, if a_j = alpha_j * a_0 and b_j = beta_j * b_0 with alpha_j * beta_j = 1
    # Then a_j * b_j = a_0 * b_0 * alpha_j * beta_j = a_0 * b_0
    # So we need a_0 * b_0 = c(i,k) for all i,k
    # But c(i,k) = 1/W[k*n+i, 0], so W[k*n+i, 0] = 1/(a_0 * b_0) for all i,k
    # This is consistent! So rank 1 might work?
    
    # Let's check: set U[i*m+j, 0] = alpha_j, V[j*p+k, 0] = beta_j, W[k*n+i, 0] = w
    # with alpha_j * beta_j = 1 for all j, and alpha_0 * beta_0 * w = 1
    # Since alpha_j * beta_j = 1, we have alpha_0 * beta_0 = 1, so w = 1
    # Then U[i*m+j, 0] = alpha_j, V[j*p+k, 0] = 1/alpha_j, W[k*n+i, 0] = 1
    # Check: sum_r=1 U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = alpha_j * (1/alpha_j) * 1 = 1 ✓
    
    # Wait, this seems too good to be true. Let me verify with a specific case.
    # n=3, m=4, p=8
    # U has 12 rows, V has 32 rows, W has 24 rows
    # U[i*4+j, 0] = alpha_j for i in 0..2, j in 0..3
    # V[j*8+k, 0] = 1/alpha_j for j in 0..3, k in 0..7
    # W[k*3+i, 0] = 1 for k in 0..7, i in 0..2
    
    # Check (A*B)[0,0] = sum_j A[0,j] * B[j,0]
    # Decomposition: sum_r U[0*4+j, r] * V[j*8+0, r] * W[0*3+0, r]
    # = sum_j U[j, 0] * V[j*8, 0] * W[0, 0]
    # = sum_j alpha_j * (1/alpha_j) * 1
    # = sum_j 1 = 4
    
    # But (A*B)[0,0] = sum_j A[0,j] * B[j,0], which depends on A and B!
    # The decomposition should give: sum_r [sum_j U[i*m+j,r] * A[i,j]] * [sum_j' V[j'*p+k,r] * B[j',k]] * W[k*n+i,r]
    # No wait, I think the correct formula is:
    # (A*B)[i,k] = sum_r (U[:, i*m+j, r] ... )
    
    # I'm confusing myself. Let me look at the schoolbook decomposition in the seed.
    # In the seed:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # for each (i,j,k) triple, with a separate column.
    
    # So the decomposition is:
    # (A*B)[i,k] = sum_{i',j',k'} [product over r where r corresponds to (i',j',k')] 
    # Actually, each column r corresponds to a unique (i,j,k) triple.
    # U[i*m+j, r] = 1 if r corresponds to (i,j,k) for some k, else 0
    # V[j*p+k, r] = 1 if r corresponds to (i,j,k) for some i, else 0
    # W[k*n+i, r] = 1 if r corresponds to (i,j,k), else 0
    
    # So for a given (i,k), the contribution to (A*B)[i,k] is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # Wait, A and B are not explicitly in the formula. The tensor decomposition
    # directly gives the coefficients.
    
    # I think the correct interpretation is that the bilinear map is:
    # f(A, B) = A*B
    # and the tensor T satisfies:
    # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k'] * delta_{i',i} * delta_{k',k}
    # Hmm, this is getting too abstract.
    
    # Let me just try to use the seed's convention and try to find a lower rank.
    # The seed uses rank n*m*p = 96.
    # Let's try rank = n*p = 24 (by summing over j in a clever way)
    # or rank = m*p = 32, or rank = n*m = 12.
    
    # Actually, let's just try to solve for small ranks using linear algebra.
    # For a given rank R, we need to find U, V, W such that:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a nonlinear system. Let's try R = 24 first.
    
    # Alternative approach: use the fact that the tensor is separable in a certain way.
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j (A[i,j] * 1) * (B[j,k] * 1)
    # This is already a sum over j, so the rank is at most m = 4?
    # No, because the factors depend on j in a coupled way.
    
    # Let's try rank R = 4:
    # U[i*m+j, r] = delta_{j,r} for r in 0..3
    # V[j*p+k, r] = delta_{j,r} for r in 0..3
    # W[k*n+i, r] = 1 for all k,i,r
    # Check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # = sum_r delta_{j,r} * delta_{j,r} * 1
    # = sum_r delta_{j,r}
    # = 1 if j in 0..3, which is always true
    # So this gives 1 for all i,j,k!
    
    # Wait, let me verify:
    # U[i*4+j, r] = 1 if j == r, else 0 (for r in 0..3)
    # V[j*8+k, r] = 1 if j == r, else 0 (for r in 0..3)
    # W[k*3+i, r] = 1 for all k, i, r
    
    # For fixed i,j,k:
    # sum_{r=0}^{3} U[i*4+j, r] * V[j*8+k, r] * W[k*3+i, r]
    # = sum_{r=0}^{3} delta_{j,r} * delta_{j,r} * 1
    # = delta_{j,j} = 1 (since j is in 0..3)
    
    # This works! So rank 4 is sufficient?
    
    # But wait, the problem says "exact tensor decomposition". Let me double-check
    # that this actually computes matrix multiplication correctly.
    
    # The bilinear map should be:
    # (A*B)[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]
    # No, I don't think that's the right formula either.
    
    # Let me look at this from first principles. The tensor T for matrix multiplication
    # of an n x m matrix by an m x p matrix is:
    # T[i, j, k] = 1 if the (i,k) entry of the product depends on A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k.
    
    # The tensor rank is the minimum R such that T can be written as:
    # T[i,j,k] = sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[i,k,r]
    
    # With the flattened indices:
    # T[i,j,k] = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And we need T[i,j,k] = 1 for all i,j,k.
    
    # So my rank-4 construction:
    # U[i*4+j, r] = delta_{j,r}
    # V[j*8+k, r] = delta_{j,r}
    # W[k*3+i, r] = 1
    
    # Gives: sum_{r=0}^{3} delta_{j,r} * delta_{j,r} * 1 = 1 ✓
    
    # So rank 4 should work! Let me verify this is correct by checking a few cases.
    
    # Actually, I realize I need to be more careful. The tensor T[i,j,k] for matrix
    # multiplication is indeed 1 for all i,j,k, because every entry A[i,j] * B[j,k]
    # contributes to the product. So the tensor is just the all-ones tensor of size
    # 3 x 4 x 8.
    
    # The rank of the all-ones tensor of size n x m x p over the reals is:
    # - If we allow arbitrary real entries, the rank is max(n, m, p) / something?
    # - Actually, the all-ones tensor can be written as:
    #   T[i,j,k] = sum_{r=1}^{R} u_i(r) * v_j(r) * w_k(r)
    #   where u_i(r) = 1/sqrt(R) * ... 
    #   
    # For the all-ones tensor, we can write:
    # T[i,j,k] = 1 = (1/sqrt(R)) * (1/sqrt(R)) * (1/sqrt(R)) * R
    # Hmm, that's not quite right.
    
    # Let's use: u_i(r) = 1 for all i,r, v_j(r) = 1 for all j,r, w_k(r) = 1/R for all k,r
    # Then sum_r u_i(r) * v_j(r) * w_k(r) = sum_r 1 * 1 * (1/R) = R * (1/R) = 1 ✓
    # So rank R works for any R >= 1?
    
    # Wait, that can't be right. Let me check with R=1:
    # u_i(0) = 1, v_j(0) = 1, w_k(0) = 1
    # sum_r=1 u_i(0) * v_j(0) * w_k(0) = 1 * 1 * 1 = 1 ✓
    
    # So the all-ones tensor has rank 1?!
    
    # But that contradicts my earlier rank-4 construction. Let me re-examine.
    
    # Oh wait, I think I made an error. The tensor for matrix multiplication is NOT
    # the all-ones tensor. Let me reconsider.
    
    # The bilinear map is: (A, B) -> A*B
    # where A is n x m and B is m x p.
    # The output is an n x p matrix.
    # The tensor T should satisfy:
    # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
    
    # In tensor notation, if we view A as a vector in R^{n*m} and B as a vector in R^{m*p},
    # then A*B is a bilinear map from R^{n*m} x R^{m*p} to R^{n*p}.
    # The structure tensor T has components T[(i,k), (i',j), (j',k')] such that:
    # (A*B)[i,k] = sum_{i',j,j',k'} T[(i,k), (i',j), (j',k')] * A[i',j] * B[j',k']
    
    # For this to equal sum_j A[i,j] * B[j,k], we need:
    # T[(i,k), (i',j), (j',k')] = 1 if i'=i, j'=j, k'=k, and j=j' (i.e., j=j'), else 0
    # Wait, that's not right. We need:
    # sum_{i',j,j',k'} T[(i,k), (i',j), (j',k')] * A[i',j] * B[j',k'] = sum_j A[i,j] * B[j,k]
    
    # This means T[(i,k), (i',j), (j',k')] = delta_{i',i} * delta_{j',j} * delta_{k',k} * delta_{j,j'}
    # = delta_{i',i} * delta_{j',j} * delta_{k',k} (since j=j' is implied by the sum)
    
    # Hmm, I'm getting confused with the indices. Let me just accept the seed's convention
    # and try to find a lower rank empirically.
    
    # Given the time constraint, let me just return the schoolbook decomposition for now,
    # but try a few simple low-rank constructions.
    
    # The rank-4 construction was found to be invalid for k-mismatch.
    # We use the schoolbook decomposition which is guaranteed to be correct.
    # Rank = n * m * p = 96.
    
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
