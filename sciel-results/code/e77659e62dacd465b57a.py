"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Trivial solution as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1
                V_trivial[j * p + k, col] = 1
                W_trivial[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        return mat.tolist()

    # Target tensor T
    # T[i, j, k] = 1 if i==0, j==0, k==0? No.
    # Matrix multiplication C = A * B.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # So T_{i, j, k} = 1 if the indices match the contraction pattern.
    # In the decomposition sum_r U_{i, r} V_{j, r} W_{k, r} = C_{i, k},
    # the term U_{i, r} V_{j, r} W_{k, r} contributes to C_{i, k} via A_{i, j} B_{j, k}.
    # Thus T_{i, j, k} should be 1 if this term is part of the sum for C_{i,k}.
    # Wait, the standard tensor for matmul is T_{i, j, k} = 1 if the indices are "compatible" in the trace.
    # Actually, the tensor T is defined such that <T, A \otimes B> = C.
    # T_{i, j, k} = 1 if i, j, k form a valid contraction?
    # Let's check the trivial solution.
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    # Sum over r (col): U[i*m+j, col] V[j*p+k, col] W[k*n+i, col]
    # This is non-zero only if col corresponds to the triplet (i, j, k).
    # In that case, the product is 1.
    # So the decomposed tensor D_{i, j, k} = 1 for all i, j, k?
    # No, the trivial solution has rank n*m*p.
    # D_{i, j, k} = sum_r U_{i, r} V_{j, r} W_{k, r}.
    # If U, V, W are as above, D_{i, j, k} = 1 for all i, j, k?
    # Let's check: For a specific (i, j, k), there is exactly one r (col) where U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    # So D_{i, j, k} = 1 for all i, j, k.
    # But the matrix multiplication tensor is NOT all ones.
    # The matrix multiplication tensor T_{i, j, k} is 1 if the indices allow contraction?
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # So the tensor T should have T_{i, j, k} = 1 for ALL i, j, k?
    # Yes, because every A_{i, j} and B_{j, k} pair contributes to some C_{i, k}.
    # Specifically, T_{i, j, k} = 1 for all i in [0, n), j in [0, m), k in [0, p).
    # Let's verify this.
    # If T is all ones, then the trivial solution with rank n*m*p works.
    # Is there a lower rank solution for the all-ones tensor?
    # The all-ones tensor of size n x m x p has rank min(n, m, p) if we are talking about standard tensor rank?
    # No, the tensor rank of the all-ones tensor is 1?
    # T_{i, j, k} = 1 = 1 * 1 * 1.
    # So U_{i, 1} = 1, V_{j, 1} = 1, W_{k, 1} = 1.
    # Then sum_r U_{i, r} V_{j, r} W_{k, r} = 1 * 1 * 1 = 1.
    # This gives rank 1!
    
    # BUT, the task says "Matrix multiplication 3,5,6".
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor T_{i, j, k} for matmul is defined such that:
    # C_{i, k} = sum_{j} A_{i, j} B_{j, k}.
    # In tensor notation, C = T(A, B).
    # T_{i, j, k} A_{i, j} B_{j, k} = C_{i, k}? No, indices don't match.
    # The tensor T has indices (i, j, k) where i is row of A, j is col of A / row of B, k is col of B.
    # The product A_{i, j} B_{j, k} contributes to C_{i, k}.
    # So T_{i, j, k} = 1 for all valid i, j, k.
    # Wait, if T is all ones, then the rank is 1.
    # Why is the trivial rank n*m*p?
    # Ah, the decomposition is for the *operation*, not just the tensor values.
    # The tensor T_{i, j, k} is 1 if the indices are "connected" in the matmul graph?
    # No, the tensor T is simply the coefficient of the monomial A_{i, j} B_{j, k} in the output C_{i, k}.
    # Since each C_{i, k} is a sum of terms, and each term is a product of one A and one B,
    # the tensor T_{i, j, k} is 1 if the term A_{i, j} B_{j, k} appears in the sum for C_{i, k}.
    # It does, for all i, j, k.
    # So T is indeed the all-ones tensor.
    # Why does the seed program use rank 90?
    # Maybe I am misunderstanding the tensor definition.
    
    # Let's re-read the seed code.
    # T_target[a * m + b, b * p + c, c * n + a] = 1.0
    # Here indices are flattened.
    # i = a, j = b, k = c.
    # The seed sets T_target to 1 ONLY at specific positions?
    # No, the loop is:
    # for a in range(n):
    #   for b in range(m):
    #     for c in range(p):
    #       T_target[a * m + b, b * p + c, c * n + a] = 1.0
    # This sets EVERY entry to 1.0.
    # So the target tensor IS the all-ones tensor.
    # If the target tensor is all ones, the rank is 1.
    # U = ones(n*m, 1), V = ones(m*p, 1), W = ones(p*n, 1).
    # Then sum_r U V W = 1 for all i, j, k.
    # So the answer should be rank 1.
    
    # Why is the seed program score 1.0 (rank 90)?
    # Maybe the evaluator checks for *integer* entries and the trivial solution is the only one found?
    # Or maybe my understanding of "rank" is wrong?
    # "Find a rank-minimal exact tensor decomposition".
    # If the tensor is all ones, rank is 1.
    # Let's check if rank 1 is allowed.
    # U, V, W must have integer/half-integer entries.
    # 1 is an integer.
    # So rank 1 should be valid.
    
    # Let's test this hypothesis.
    # If I return rank 1, the score should be 90/1 = 90.
    # This would be a massive improvement.
    
    # Why did the seed program fail?
    # It tried to find a rank 60 solution with random search.
    # It didn't try rank 1.
    
    # Is there a catch?
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # This is just indexing.
    
    # Let's double check the tensor definition.
    # Matrix multiplication: C = AB.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T is such that C_{ik} = sum_{j} T_{ijk} A_{ij} B_{jk}?
    # No, the standard tensor definition is T_{ijk} where the term is A_{ij} B_{jk} W_{ki}?
    # No, the decomposition is sum_r U_{ir} V_{jr} W_{kr} = T_{ijk}.
    # And we want this to represent the matmul.
    # The "value" of the tensor at (i, j, k) is 1 if the product A_{ij} B_{jk} contributes to C_{ik}.
    # It does.
    # So T_{ijk} = 1 for all i, j, k.
    
    # Therefore, the rank is 1.
    
    # Let's construct the rank 1 solution.
    U_rank1 = np.ones((n * m, 1))
    V_rank1 = np.ones((m * p, 1))
    W_rank1 = np.ones((p * n, 1))
    
    # Check exactness
    T_approx = np.einsum('ir,jr,kr->ijk', U_rank1, V_rank1, W_rank1)
    T_target = np.ones((n, m, p))
    
    if np.allclose(T_approx, T_target):
        return {
            "U": U_rank1.tolist(),
            "V": V_rank1.tolist(),
            "W": W_rank1.tolist()
        }
        
    # Fallback to trivial if for some reason rank 1 is rejected (unlikely)
    return {
        "U": to_list(U_trivial),
        "V": to_list(V_trivial),
        "W": to_list(W_trivial)
    }
# EVOLVE-BLOCK-END
