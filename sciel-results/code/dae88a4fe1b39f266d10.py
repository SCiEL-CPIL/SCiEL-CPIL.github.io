"""Matrix multiplication 3x4x7 - subspace peeling approach."""
DIMENSIONS = [3, 4, 7]

from fractions import Fraction
import numpy as np
from scipy.linalg import null_space

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Subspace peeling: iteratively find rank-1 components that can be peeled off
    # The matmul tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We seek T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Try to find a decomposition with rank < n*m*p
    # For 3x4x7, let's try a structured approach.
    
    # One known technique: if we can factor the index spaces, we can reduce rank.
    # Here n=3, m=4, p=7. None of these share obvious factors that help.
    
    # Let's try the "peeling" approach: 
    # 1. Start with the full tensor T (all ones)
    # 2. Find a rank-1 approximation u ⊗ v ⊗ w that is "aligned" with T
    # 3. Subtract and continue on the residual
    
    # Actually, for exact decompositions of the matmul tensor, a common approach is:
    # Use the fact that C[i,k] = sum_j A[i,j] B[j,k]
    # Group by j: C[i,k] = sum_j (A[i,j] * B[j,k])
    # This gives rank = m * (number of independent (i,k) pairs) but that's still n*m*p.
    
    # Better: For fixed i, C[i,:] = A[i,:] * B, which is a row of A times B.
    # For fixed k, C[:,k] = A * B[:,k], which is A times a column of B.
    
    # Let me try a different approach: use the structure to find a lower rank.
    # 
    # Key insight: The matmul tensor has a "slice" structure.
    # Slice along j: T_j[i,k] = 1 for all i,k. All slices are identical!
    # This means T = sum_j e_j ⊗ (all-ones_{n x p})
    # But that's rank m * (rank of all-ones matrix) = m * 1 = m? No, that's not right
    # because we need the full 3-way tensor.
    
    # Actually T[i,j,k] = 1 for all i,j,k. This is just the all-ones tensor!
    # Wait, no. The matmul tensor is T[i,j,k] = 1 for all valid (i,j,k).
    # Yes, it IS the all-ones tensor of shape (n,m,p).
    
    # The all-ones tensor of shape (n,m,p) has rank 1! Because:
    # T[i,j,k] = 1 * 1 * 1 = u_i * v_j * w_k where u,v,w are all-ones vectors.
    
    # So the rank of the 3x4x7 matmul tensor is 1?!
    # That can't be right. Let me re-examine.
    
    # Oh wait, I think I'm confusing the structure. The matrix multiplication tensor
    # is NOT the all-ones tensor. Let me reconsider.
    
    # The structure tensor of matrix multiplication (n x m) * (m x p) -> (n x p)
    # has T_{i,j,k} = 1 if the (i,k) entry of the product depends on A_{i,j} and B_{j,k}.
    # Since C_{i,k} = sum_j A_{i,j} B_{j,k}, every (i,j,k) triple contributes.
    # So T_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    
    # Hmm, but that would make it rank 1, which is clearly wrong since we know
    # matmul requires rank at least max(nm, mp, np) = 28.
    
    # I think the issue is in how the tensor decomposition is defined.
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must equal 1 for all (i,j,k) to "reproduce" the matmul.
    
    # But wait, that's not right either. The tensor T[i,j,k] represents the
    # coefficient in the bilinear computation. For matmul, T[i,j,k] = 1.
    # And we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k] = 1
    # for all (i,j,k).
    
    # If T is the all-ones tensor, then rank 1 works: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1.
    # But that would mean we can compute C[i,k] = A[i,j]*B[j,k] for a SINGLE (i,j,k),
    # not the sum over j. So I'm misunderstanding the tensor definition.
    
    # Let me reconsider: The matmul tensor should encode that C[i,k] = sum_j A[i,j]*B[j,k].
    # The "tensor" here is not T[i,j,k]=1 for all triples. Rather, the bilinear form is:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # The tensor of the bilinear map (A,B) -> C is a 3-tensor where:
    # The coefficient of A[i,j] * B[j,k] in C[i',k'] is delta_{i,i'} * delta_{k,k'}
    # So T[i',j,k',i,j,k]... no, this is getting complicated.
    
    # Let me look at it differently. The decomposition:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must hold for ALL (i,j,k) such that the product A[i,j]*B[j,k] contributes to C[i,k].
    # 
    # Actually, I think the correct interpretation is:
    # For each r, we have a "term" that computes a partial product.
    # The full computation is: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # No, that's not right either.
    
    # Let me just go with the schoolbook decomposition which is definitely correct,
    # and try to see if we can find a better one through search.
    
    # Schoolbook: rank = n*m*p = 84
    # For each (i,j,k), one term with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1
    
    # Let me try to verify if a lower rank is possible by checking known results.
    # For 2x2x2 matmul, rank is 7 (Strassen).
    # For 3x3x3, rank is at most 23 (Bini et al.), lower bound 19.
    # For 3x4x7, I'm not sure of the exact minimal rank.
    
    # Let me try a heuristic: use the schoolbook but try to merge terms.
    # Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if they share structure.
    
    # Actually, let me just try to find a rank-28 decomposition using the lower bound.
    # The lower bound is max(nm, mp, np) = max(12, 21, 28) = 28.
    
    # One approach: For each k (0..6), we need to compute C[:,k] = A * B[:,k].
    # This is a 3x7 vector = 3x4 matrix * 4x1 vector.
    # The rank of this matvec is at most 3*4 = 12 per k, but we can share.
    
    # Let me try: rank = m * n = 4 * 3 = 12? No, that's too low.
    # rank = m * p = 4 * 7 = 28? Let's try this.
    
    # For each (j,k) pair, define a term. But we need to sum over j for each (i,k).
    # Term for (j,k): U[i*m+j, r_{j,k}] = A[i,j] coefficient... but U should be constants!
    
    # I see the issue: U, V, W are FIXED matrices (constants), not depending on A or B.
    # The decomposition is: for each r, we compute an intermediate value,
    # and the final C[i,k] = sum_r W[k*n+i, r] * (intermediate_r)
    # where intermediate_r involves U and V.
    
    # Specifically: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # No, that's a quadratic form, not bilinear.
    
    # I think the correct form is:
    # C[i,k] = sum_r W[k*n+i, r] * U[i*m+?, r] * V[?*p+k, r]
    # But U and V indices must depend on i,j and j,k respectively.
    
    # I think the standard form is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # No...
    
    # Let me just keep the schoolbook. It's correct and gives rank 84.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
